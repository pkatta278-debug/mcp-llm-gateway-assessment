"""Task 1 - Custom MCP server with strict validation over stdio.

Tools
-----
get_customer_record(customer_id)             customer_id must match CUST-XXXXX (5 digits)
trigger_refund(customer_id, amount, reason)  amount > 0, reason >= 10 chars

Design rules
------------
* stdout is reserved for JSON-RPC frames. Nothing in this file calls print();
  logging is configured to write to stderr only.
* Malformed input is a *protocol* error: it is raised as MCPError(INVALID_PARAMS,
  -32602) and the SDK turns it into a JSON-RPC error response.
* A well-formed call that cannot be completed (unknown customer, refund over the
  limit) is a *tool execution* error: it comes back as a normal result with
  isError=true, as the MCP spec prescribes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from typing import Annotated, Any

import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

# --------------------------------------------------------------------------- #
# Logging: stderr only. Never stdout.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [task1-mcp] %(message)s",
)
log = logging.getLogger("task1-mcp")

SERVER_NAME = "customer-ops"
SERVER_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Input schemas (Pydantic = single source of truth for validation AND the
# inputSchema advertised through tools/list)
# --------------------------------------------------------------------------- #
# [0-9] on purpose: \d also matches non-ASCII digits (e.g. Arabic-Indic).
# The pattern is anchored by Pydantic's engine, so "CUST-12345\n" is rejected.
CustomerId = Annotated[
    str,
    StringConstraints(pattern=r"^CUST-[0-9]{5}$"),
    Field(description="Customer ID in the format CUST-XXXXX (five digits).", examples=["CUST-10001"]),
]


class GetCustomerRecordArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: CustomerId


class TriggerRefundArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: CustomerId
    amount: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Refund amount, positive, at most 2 decimal places.",
    )
    reason: str = Field(
        min_length=10,
        max_length=500,
        description="Why the refund is being issued (at least 10 characters).",
    )

    @field_validator("amount")
    @classmethod
    def _whole_cents_only(cls, v: float) -> float:
        # Money can't be split below one cent.
        if round(v, 2) != v:
            raise ValueError("amount must have at most 2 decimal places")
        return v

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, v: str) -> str:
        # "          " (10 spaces) satisfies min_length but is not a reason.
        if len(v.strip()) < 10:
            raise ValueError("reason must contain at least 10 non-whitespace-padded characters")
        return v


# --------------------------------------------------------------------------- #
# Mock data (stand-in for a real CRM / billing backend)
# --------------------------------------------------------------------------- #
CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-10001": {"name": "Asha Verma", "email": "asha.verma@example.com", "tier": "gold", "status": "active", "total_paid": 1250.00},
    "CUST-10002": {"name": "Rohan Mehta", "email": "rohan.mehta@example.com", "tier": "silver", "status": "active", "total_paid": 480.50},
    "CUST-10003": {"name": "Neha Singh", "email": "neha.singh@example.com", "tier": "bronze", "status": "suspended", "total_paid": 99.99},
}
REFUNDS: list[dict[str, Any]] = []
_refund_lock = asyncio.Lock()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _text_result(payload: dict[str, Any], *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        is_error=is_error,
    )


def _invalid_params(tool: str, exc: ValidationError) -> MCPError:
    # Deliberately drop pydantic's "input" echo: never reflect raw user data back.
    problems = [
        {
            "field": ".".join(str(p) for p in err["loc"]) or "(arguments)",
            "type": err["type"],
            "message": err["msg"],
        }
        for err in exc.errors(include_input=False, include_url=False, include_context=False)
    ]
    return MCPError(
        code=types.INVALID_PARAMS,
        message=f"Invalid params for tool '{tool}'",
        data={"errors": problems},
    )


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
async def _get_customer_record(args: GetCustomerRecordArgs) -> types.CallToolResult:
    customer = CUSTOMERS.get(args.customer_id)
    if customer is None:
        return _text_result({"error": "customer_not_found", "customer_id": args.customer_id}, is_error=True)
    public = {k: v for k, v in customer.items() if k != "total_paid"}
    return _text_result({"customer_id": args.customer_id, **public})


async def _trigger_refund(args: TriggerRefundArgs) -> types.CallToolResult:
    async with _refund_lock:  # serialise check-then-write on the ledger
        customer = CUSTOMERS.get(args.customer_id)
        if customer is None:
            return _text_result({"error": "customer_not_found", "customer_id": args.customer_id}, is_error=True)
        if customer["status"] != "active":
            return _text_result(
                {"error": "customer_not_eligible", "customer_id": args.customer_id, "status": customer["status"]},
                is_error=True,
            )
        already = sum(r["amount"] for r in REFUNDS if r["customer_id"] == args.customer_id)
        remaining = round(customer["total_paid"] - already, 2)
        if args.amount > remaining:
            return _text_result(
                {"error": "refund_exceeds_total_paid", "customer_id": args.customer_id, "max_refundable": remaining},
                is_error=True,
            )
        refund = {
            "refund_id": f"RFD-{uuid.uuid4().hex[:8].upper()}",
            "customer_id": args.customer_id,
            "amount": args.amount,
            "reason": args.reason,
            "status": "queued",
        }
        REFUNDS.append(refund)
    return _text_result({k: v for k, v in refund.items() if k != "reason"})


# name -> (description, args model, implementation)
TOOLS: dict[str, tuple[str, type[BaseModel], Any]] = {
    "get_customer_record": (
        "Fetch a customer's profile by customer ID (format CUST-XXXXX).",
        GetCustomerRecordArgs,
        _get_customer_record,
    ),
    "trigger_refund": (
        "Queue a refund for an active customer. Requires a positive amount and a reason of at least 10 characters.",
        TriggerRefundArgs,
        _trigger_refund,
    ),
}


# --------------------------------------------------------------------------- #
# MCP handlers
# --------------------------------------------------------------------------- #
async def dispatch_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
    """Validate and run one tool call. Pure function of (name, arguments): easy to unit test."""
    entry = TOOLS.get(name)
    if entry is None:
        # MCP spec: unknown tool is a protocol error, -32602.
        raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {name}")
    _, model, impl = entry

    try:
        args = model.model_validate({} if arguments is None else arguments)
    except ValidationError as exc:
        log.warning("tools/call %s rejected: %d validation error(s)", name, exc.error_count())
        raise _invalid_params(name, exc) from None

    result = await impl(args)
    log.info("tools/call %s ok (isError=%s)", name, result.is_error)
    return result


async def on_list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(name=name, description=desc, input_schema=model.model_json_schema())
            for name, (desc, model, _) in TOOLS.items()
        ]
    )


async def on_call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> types.CallToolResult:
    return await dispatch_tool(params.name, params.arguments)


def build_server() -> Server:
    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def main() -> None:
    server = build_server()
    log.info("starting %s %s on stdio", SERVER_NAME, SERVER_VERSION)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    log.info("stdin closed, shutting down")


if __name__ == "__main__":
    asyncio.run(main())

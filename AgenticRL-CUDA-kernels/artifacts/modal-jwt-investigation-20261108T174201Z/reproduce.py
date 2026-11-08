import argparse
import asyncio
import base64
import hashlib
import json
import platform
from importlib.metadata import distribution, version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from modal import _functions
from modal._utils import auth_token_manager as auth
from modal.client import _Client
from modal.exception import AuthError
from modal_proto import api_pb2


def synthetic_token(expiry):
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=")
    return f"synthetic.{payload}.unsigned"


async def manager_case(name, second_time, second_result):
    clock = {"now": 1000.0}
    warnings = []
    calls = []
    old_token = synthetic_token(1100.0)
    new_token = synthetic_token(1200.0)

    async def fetch(retry):
        calls.append(retry)
        if len(calls) == 1:
            return old_token
        if second_result == "transient":
            raise RuntimeError("synthetic_transient_refresh_failure")
        if second_result == "denied":
            raise AuthError("synthetic_auth_denied")
        return new_token

    with patch.object(auth, "time", SimpleNamespace(time=lambda: clock["now"])), \
         patch.object(auth, "random", SimpleNamespace(random=lambda: 0.0)), \
         patch.object(auth, "logger", SimpleNamespace(warning=lambda *args: warnings.append(args[0]))):
        manager = auth._AuthTokenManager(SimpleNamespace(), fetch=fetch)
        assert await manager.get_token() == old_token
        await asyncio.sleep(0)
        clock["now"] = second_time
        error = None
        returned = None
        try:
            returned = await manager.get_token()
        except AuthError:
            error = "AuthError"
        assert len(calls) == 2
        report = {"scenario": name, "client_time": clock["now"], "old_expiry": 1100.0,
                  "fetch_calls": len(calls), "error": error,
                  "returned_cached": returned == old_token,
                  "returned_expired_token": returned is not None and manager._is_expired(),
                  "warning_count": len(warnings)}
        if second_result == "denied":
            assert error == "AuthError" and returned is None
        elif second_result == "success":
            assert returned == new_token and not manager._is_expired()
        else:
            assert returned == old_token and len(warnings) == 1
            assert manager._is_expired() == (second_time >= 1100.0)
        return report


async def missing_token_case():
    async def fetch(retry):
        raise RuntimeError("synthetic_transient_fetch_failure")

    manager = auth._AuthTokenManager(SimpleNamespace(), fetch=fetch)
    try:
        await manager.get_token()
    except RuntimeError:
        assert manager._token == ""
        return {"scenario": "no_cached_token_fetch_failure", "error": "RuntimeError",
                "returned_token": False}
    raise AssertionError("Missing-token fetch failure was concealed")


async def invocation_case(server_rejects):
    clock = {"now": 1000.0}
    fetch_calls = []
    requests = []
    headers = []

    async def fetch(retry):
        fetch_calls.append(retry)
        return synthetic_token(1100.0 if len(fetch_calls) == 1 else 1300.0)

    manager = auth._AuthTokenManager(SimpleNamespace(), fetch=fetch)

    class FakeClient:
        _auth_token_manager = manager
        _stub = SimpleNamespace()

        async def get_input_plane_metadata(self, region):
            return await _Client.get_input_plane_metadata(self, region)

    class FakeStub:
        async def AttemptAwait(self, request, *, retry, metadata):
            requests.append(request)
            headers.append(dict(metadata)["x-modal-auth-token"])
            if server_rejects:
                raise AuthError("Jwt is expired")
            if len(requests) == 1:
                clock["now"] = 1101.0
                return api_pb2.AttemptAwaitResponse()
            return api_pb2.AttemptAwaitResponse(output=api_pb2.FunctionGetOutputsItem(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)))

    invocation = _functions._InputPlaneInvocation(
        FakeStub(), "synthetic-attempt-opaque", FakeClient(),
        api_pb2.FunctionPutInputsItem(), "fu-synthetic",
        api_pb2.FunctionRetryPolicy(retries=0), "synthetic-region")
    with patch.object(auth, "time", SimpleNamespace(time=lambda: clock["now"])), \
         patch.object(auth, "random", SimpleNamespace(random=lambda: 0.0)), \
         patch.object(_functions, "_process_result", AsyncMock(return_value="synthetic-done")):
        if server_rejects:
            try:
                await invocation.run_function()
            except AuthError as exc:
                assert str(exc) == "Jwt is expired"
            else:
                raise AssertionError("AttemptAwait AuthError was concealed")
            assert len(requests) == 1 and len(fetch_calls) == 1 and not manager._is_expired()
            return {"scenario": "synthetic_rpc_auth_error_with_locally_valid_metadata",
                    "await_calls": 1, "fetch_calls": 1, "locally_valid_metadata": True,
                    "error": "AuthError: Jwt is expired", "automatic_auth_retry": False}
        assert await invocation.run_function() == "synthetic-done"
        assert len(requests) == 2 and len(fetch_calls) == 2
        assert headers[0] != headers[1]
        assert requests[0].attempt_token == requests[1].attempt_token
        assert not manager._is_expired()
        return {"scenario": "sync_await_refreshes_expired_metadata_between_polls",
                "await_calls": 2, "fetch_calls": 2, "header_changed": True,
                "attempt_token_unchanged": True, "completed": True}


async def reproduce():
    assert version("modal") == "1.6.3"
    cases = [
        await missing_token_case(),
        await manager_case("valid_cached_proactive_refresh_failure", 1060.0, "transient"),
        await manager_case("expired_cached_transient_refresh_failure", 1101.0, "transient"),
        await manager_case("expired_cached_auth_denied", 1101.0, "denied"),
        await manager_case("expired_cached_successful_refresh", 1101.0, "success"),
        await invocation_case(False),
        await invocation_case(True),
    ]
    bug = next(case for case in cases if case["scenario"] == "expired_cached_transient_refresh_failure")
    assert bug["returned_expired_token"] and bug["error"] is None
    dist = distribution("modal")
    records = {str(path): path for path in dist.files or []}
    sources = {}
    for name in ("modal/_utils/auth_token_manager.py", "modal/_functions.py",
                 "modal/client.py", "modal/_grpc_client.py", "modal/_utils/logger.py"):
        record = records[name]
        raw = Path(dist.locate_file(record)).read_bytes()
        digest = hashlib.sha256(raw).digest()
        encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        assert record.hash.mode == "sha256" and encoded == record.hash.value, name
        sources[name] = {"sha256": digest.hex(), "installed_record_matches": True}
    return {"status": "reproductions_passed", "modal_version": version("modal"),
            "python_version": platform.python_version(), "network_calls": 0,
            "real_tokens_used": False, "historical_cause_established": False,
            "confirmed_sdk_defect": "Expired cached token is returned after a non-auth refresh failure.",
            "cases": cases, "installed_sdk_sources": sources,
            "limitations": [
                "These are offline real-SDK control-flow reproductions with synthetic tokens/RPCs.",
                "No server JWT validation or real refresh outage was reproduced.",
                "The historical log does not identify whether header or attempt JWT expired.",
                "A shared error string or stack frame does not prove which mechanism occurred.",
            ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(reproduce())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

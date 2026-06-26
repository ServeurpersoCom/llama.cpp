import json
import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_scope = True


def open_stream(query: str = "") -> requests.Response:
    url = f"http://{server.server_host}:{server.server_port}/scope/stream{query}"
    return requests.get(url, stream=True, timeout=30)


def read_frames(response: requests.Response, n: int) -> list:
    # keepalive comments count toward the budget so a quiet stream terminates the read
    frames = []
    budget = 30
    for line_bytes in response.iter_lines():
        line = line_bytes.decode("utf-8")
        if line.startswith("data: "):
            frames.append(json.loads(line[6:]))
            if len(frames) >= n:
                break
        budget -= 1
        if budget <= 0:
            break
    return frames


def stream_during_completion(query: str = "", n_frames: int = 2) -> list:
    res = open_stream(query)
    assert res.status_code == 200
    cmpl = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 4,
    })
    assert cmpl.status_code == 200
    frames = read_frames(res, n_frames)
    res.close()
    return frames


def test_scope_stream():
    global server
    server.start()
    frames = stream_during_completion(n_frames=3)
    assert len(frames) == 3
    for i, frame in enumerate(frames):
        # one frame per decode, seq is monotonic from zero
        assert frame["seq"] == i
        assert frame["n_nodes"] > 0
        assert frame["n_leafs"] > 0
        nodes = frame["nodes"]
        assert len(nodes) == frame["n_nodes"]
        # every src entry references an earlier position, the DAG is topologically ordered
        for j, node in enumerate(nodes):
            for s in node.get("src", []):
                assert 0 <= s < j
        # leaf entries carry a known kind
        kinds = {n["leaf"] for n in nodes if "leaf" in n}
        assert kinds <= {"weight", "kv", "input", "uncaptured"}
        # stats are present on float nodes and finite bounds are ordered
        stats = [n["stats"] for n in nodes if "stats" in n]
        assert len(stats) > 0
        for st in stats:
            if "min" in st:
                assert st["min"] <= st["max"]


def test_scope_stream_filtered():
    global server
    server.start()
    frames = stream_during_completion("?filter=ffn_out", n_frames=2)
    assert len(frames) == 2
    for frame in frames:
        compute = [n for n in frame["nodes"] if "leaf" not in n]
        assert len(compute) > 0
        for node in compute:
            assert "ffn_out" in node["name"]


def test_scope_filter_matches_nothing():
    global server
    server.start()
    # a decode still emits its frame so frames keep pairing with tokens
    frames = stream_during_completion("?filter=zzz_no_such_node", n_frames=2)
    assert len(frames) == 2
    for frame in frames:
        assert frame["n_nodes"] == 0


def test_scope_stream_exclusive():
    global server
    server.start()
    first = open_stream()
    assert first.status_code == 200
    second = open_stream()
    assert second.status_code == 503
    first.close()
    second.close()


def test_scope_release_on_disconnect():
    global server
    server.start()
    first = open_stream()
    assert first.status_code == 200
    first.close()
    # the stream slot frees once the server notices the disconnect
    deadline = time.time() + 10
    status = 503
    while time.time() < deadline:
        retry = open_stream()
        status = retry.status_code
        retry.close()
        if status == 200:
            break
        time.sleep(0.5)
    assert status == 200


def test_scope_invalid_regex():
    global server
    server.start()
    res = open_stream("?filter=%5Binvalid")
    assert res.status_code == 400
    res.close()


def test_scope_disabled():
    global server
    server.server_scope = False
    server.start()
    res = open_stream()
    assert res.status_code == 501
    res.close()

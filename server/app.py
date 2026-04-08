from __future__ import annotations
import json
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from models import ALL_TASK_IDS, CodeReviewAction
from server.environment import CodeReviewEnvironment

app = FastAPI(
    title="Code Review Environment",
    description="OpenEnv RL environment - AI agent reviews buggy Python code",
    version="1.0.0",
)

#  HTTP Endpoints 

@app.get("/health")
def health():
    return {"status": "healthy", "tasks": ALL_TASK_IDS, "version": "1.0.0"}

@app.post("/reset")
def reset(task_id: str = Query(default="easy_01", description="Task ID to load")):
    if task_id not in ALL_TASK_IDS:
        return {"error": f"Unknown task_id. Valid: {ALL_TASK_IDS}"}
    env = CodeReviewEnvironment(task_id=task_id)
    obs = env.reset()
    return obs.dict()

class StepRequest(BaseModel):
    review_text: str
    task_id: str = "easy_01"

_stateless_envs: dict[str, CodeReviewEnvironment] = {}

@app.post("/step")
def step(body: StepRequest):
    if body.task_id not in _stateless_envs:
        env = CodeReviewEnvironment(task_id=body.task_id)
        env.reset()
        _stateless_envs[body.task_id] = env
    env = _stateless_envs[body.task_id]
    action = CodeReviewAction(review_text=body.review_text)
    obs, reward, done = env.step(action)
    if done:
        _stateless_envs.pop(body.task_id, None)
    return {"observation": obs.dict(), "reward": reward, "done": done}

@app.get("/state")
def state(task_id: str = Query(default="easy_01")):
    if task_id in _stateless_envs:
        return _stateless_envs[task_id].state().dict()
    env = CodeReviewEnvironment(task_id=task_id)
    return env.reset().dict()

@app.get("/tasks")
def list_tasks():
    from models import TASK_LIBRARY
    return [
        {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "task_type": t.task_type,
            "description": t.task_description[:120] + "...",
        }
        for t in TASK_LIBRARY
    ]

#  Web UI 

@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
<title>Code Review Environment</title>
<style>
  body { font-family: monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 24px; }
  h1 { color: #58a6ff; }
  select, textarea, button { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 8px; font-family: monospace; }
  textarea { width: 100%; box-sizing: border-box; }
  button { cursor: pointer; background: #238636; border-color: #2ea043; color: #fff; padding: 10px 20px; }
  button:hover { background: #2ea043; }
  pre { background: #161b22; padding: 16px; border-radius: 6px; border: 1px solid #30363d; overflow-x: auto; white-space: pre-wrap; }
  .reward { font-size: 1.4em; color: #3fb950; font-weight: bold; }
  .label { color: #8b949e; font-size: 0.85em; margin-top: 12px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
</style>
</head>
<body>
<h1>Code Review Environment</h1>
<div class="label">Select Task</div>
<select id="taskSelect">
  <option value="easy_01">easy_01 - ZeroDivisionError bug</option>
  <option value="easy_02">easy_02 - Division by zero on b=0</option>
  <option value="medium_01">medium_01 - KeyError + typo bug</option>
  <option value="hard_01">hard_01 - SQL injection + 2 bugs</option>
  <option value="perf_01">perf_01 - O(n^2) performance</option>
  <option value="logic_01">logic_01 - Binary search 3 bugs</option>
</select>
<button id="loadTaskBtn" style="margin-left:8px">Load Task</button>
<div class="grid" style="margin-top:16px">
  <div>
    <div class="label">Code to Review</div>
    <pre id="codeBox">Click "Load Task" to begin.</pre>
  </div>
  <div>
    <div class="label">Task Description</div>
    <pre id="descBox"></pre>
  </div>
</div>
<div class="label">Your Review (write your code review here)</div>
<textarea id="reviewBox" rows="8" placeholder="Describe bugs found, their causes, severity, and fixes..."></textarea>
<br><br>
<button onclick="submitReview()">Submit Review</button>
<div class="label" style="margin-top:16px">Result</div>
<div class="reward" id="rewardBox"></div>
<pre id="resultBox"></pre>
<script>
let ws = null;

document.getElementById("loadTaskBtn").onclick = function() {
  const taskId = document.getElementById("taskSelect").value;
  
  if (ws) ws.close();
  const protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
  ws = new WebSocket(protocol + window.location.host + "/ws");
  // Keep WebSocket initialization for the step endpoint
  ws.onopen = () => ws.send(JSON.stringify({type:"reset", task_id: taskId}));
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    const obs = d.observation || d;
    if (d.reward !== undefined) {
      document.getElementById("rewardBox").textContent = "Reward: " + d.reward.toFixed(4);
      document.getElementById("resultBox").textContent = JSON.stringify(obs, null, 2);
    }
  };

  fetch(`/reset?task_id=${taskId}`, {
      method: "POST"
  })
  .then(res => res.json())
  .then(response => {
      console.log(response);
      document.getElementById("codeBox").textContent = response.code_snippet || "";
      document.getElementById("descBox").textContent = response.task_description || "";
  })
  .catch(err => console.error("Error loading task:", err));
};

function submitReview() {
  if (!ws || ws.readyState !== 1) { alert("Load a task first!"); return; }
  const review = document.getElementById("reviewBox").value;
  ws.send(JSON.stringify({type:"step", review_text: review}));
}
</script>
</body>
</html>
""")

#  WebSocket 

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env: Optional[CodeReviewEnvironment] = None
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "reset":
                task_id = data.get("task_id", "easy_01")
                env = CodeReviewEnvironment(task_id=task_id)
                obs = env.reset()
                await websocket.send_json({
                    "observation": obs.dict(), "reward": 0.0, "done": False
                })

            elif msg_type == "step" and env:
                review_text = data.get("review_text", "")
                action = CodeReviewAction(review_text=review_text)
                obs, reward, done = env.step(action)
                response = {"observation": obs.dict(), "reward": reward, "done": done}
                if done:
                    response["episode_stats"] = env.episode_stats().dict()
                await websocket.send_json(response)

            elif msg_type == "state" and env:
                await websocket.send_json({"observation": env.state().dict()})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

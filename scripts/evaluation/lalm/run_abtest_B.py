#!/usr/bin/env python3
import argparse, base64, json, time, sys, re, os
import concurrent.futures as cf
from pathlib import Path
import urllib.request

def norm_utt(u): return re.sub(r'^0+', '', str(u).strip()) or '0'
def get_audio_path(utt, search_dir):
    d = Path(search_dir)
    if not d.exists(): return None
    target = norm_utt(utt)
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in ['.wav', '.mp3', '.flac']:
            if norm_utt(p.stem) == target: return p
    return None

def call_api(enc_a, enc_b, ctx, prompt_t, api_url, api_key, model_name):
    prompt = prompt_t + "\n<script_data>\n" + json.dumps(ctx, ensure_ascii=False) + "\n</script_data>"
    payload = {"model": model_name, "max_tokens": 8192, "temperature": 0.0, "top_p": 0.1,
               "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "input_audio", "input_audio": {"data": enc_a, "format": "wav"}}, {"type": "text", "text": "Above is Audio A. Below is Audio B:"}, {"type": "input_audio", "input_audio": {"data": enc_b, "format": "wav"}}]}]}
    for att in range(3):
        try:
            req = urllib.request.Request(api_url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            res = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())["choices"][0]["message"]["content"]
            res = res[res.find('{'):res.rfind('}')+1]
            return json.loads(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', res), strict=False)
        except Exception as e:
            if att == 2: raise e; time.sleep(5)

def build_dual_ctx(row_asr, row_gt, is_swap):
    ctx = {"context_background": row_gt.get("context_background", ""), "character": row_gt.get("character", {}), "content": []}
    for gt_t in row_gt.get("content", []):
        t_num = gt_t.get("turn_num")
        asr_t = next((t for t in row_asr.get("content", []) if t.get("turn_num") == t_num), {})
        g_ts = gt_t.get("audio_assets", {})
        m_ts = asr_t.get("audio_assets", g_ts)
        turn_data = {"turn_num": t_num, "speaker_info": gt_t.get("speaker_info", {}), "text": gt_t.get("text", ""), "context": gt_t.get("context", "")}
        if is_swap: turn_data["Audio_A_timestamps_ms"], turn_data["Audio_B_timestamps_ms"] = g_ts, m_ts
        else: turn_data["Audio_A_timestamps_ms"], turn_data["Audio_B_timestamps_ms"] = m_ts, g_ts
        ctx["content"].append(turn_data)
    return ctx

def evaluate(task, args, prompt_t, gt_data_dict):
    row_asr, run_index = task
    utt = str(row_asr.get("utt"))
    row_gt = gt_data_dict.get(norm_utt(utt))
    if not row_gt: return {"utt": utt, "run_index": run_index, "status": "error", "error": "GT data missing"}
    
    path_m, path_g = get_audio_path(utt, args.audio_dir_model), get_audio_path(utt, args.audio_dir_gt)
    if not path_m or not path_g: return {"utt": utt, "run_index": run_index, "status": "error", "error": "Audio missing"}
    
    try:
        enc_m, enc_g = base64.b64encode(path_m.read_bytes()).decode("ascii"), base64.b64encode(path_g.read_bytes()).decode("ascii")
        ctx_norm = build_dual_ctx(row_asr, row_gt, is_swap=False)
        ctx_swap = build_dual_ctx(row_asr, row_gt, is_swap=True)
        
        res_norm = call_api(enc_m, enc_g, ctx_norm, prompt_t, args.api_url, args.api_key, args.model_name)
        res_swap = call_api(enc_g, enc_m, ctx_swap, prompt_t, args.api_url, args.api_key, args.model_name)
        
        resolved = []
        u_n = {u.get("turn_num", i): u for i, u in enumerate(res_norm.get("utterance_evaluations", []))}
        u_s = {u.get("turn_num", i): u for i, u in enumerate(res_swap.get("utterance_evaluations", []))}
        for t_num in u_n.keys():
            if t_num not in u_s: continue
            turn_res = {"turn_num": t_num}
            for dim in ["B1_Instruction_Following", "B2_Prosody_Fluency", "B3_Local_Expressiveness"]:
                w_n, w_s = u_n[t_num].get(dim, {}).get("winner", "Tie"), u_s[t_num].get(dim, {}).get("winner", "Tie")
                r_n = "Model" if w_n == "A" else ("GT" if w_n == "B" else "Tie")
                r_s = "GT" if w_s == "A" else ("Model" if w_s == "B" else "Tie")
                turn_res[dim] = {"final_winner": r_n if (r_n == r_s and r_n != "Tie") else "Tie", "norm_winner": r_n, "swap_winner": r_s, "reasoning": u_n[t_num].get(dim, {}).get("comparison_reasoning", "")}
            resolved.append(turn_res)
        return {"utt": utt, "run_index": run_index, "status": "ok", "result": {"utterances": resolved}}
    except Exception as e: return {"utt": utt, "run_index": run_index, "status": "error", "error": str(e)[:100]}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--audio_dir_model", required=True)
    p.add_argument("--audio_dir_gt", required=True)
    p.add_argument("--output", default="../../../results/metrics/lalm/local_eval_results.jsonl")
    p.add_argument("--run_index", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--api_url", default="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    p.add_argument("--api_key", default="")
    p.add_argument("--model_name", default="gemini-3.8-flash")
    args = p.parse_args()
    
    if not args.api_key:
        print("Error: Please provide --api_key or set the default value in the script.")
        sys.exit(1)
    
    prompt_path = Path(__file__).parent / "prompts" / "abtest_prompt_B.txt"
    prompt_t = prompt_path.read_text(encoding='utf-8')
    
    out_f = Path(args.output).resolve(); out_f.parent.mkdir(parents=True, exist_ok=True)
    done = {norm_utt(json.loads(l)["utt"]) for l in open(out_f, encoding='utf-8') if json.loads(l).get("status")=="ok"} if out_f.exists() else set()
    
    gt_data = {norm_utt(json.loads(l)["utt"]): json.loads(l) for l in open(args.gt_jsonl, encoding='utf-8') if l.strip()}
    
    tasks = []
    for l in open(args.jsonl, encoding='utf-8'):
        if not l.strip(): continue
        d = json.loads(l)
        u_norm = norm_utt(d["utt"])
        if u_norm in gt_data and u_norm not in done: 
            tasks.append((d, args.run_index))
            
    if tasks:
        with open(out_f, "a", encoding='utf-8') as f, cf.ThreadPoolExecutor(args.concurrency) as pool:
            for count, fut in enumerate(cf.as_completed([pool.submit(evaluate, t, args, prompt_t, gt_data) for t in tasks]), 1):
                res = fut.result(); f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
                print(f"[{count}/{len(tasks)}] Local Run {args.run_index} | {res['utt']} -> {res['status']}", file=sys.stderr)

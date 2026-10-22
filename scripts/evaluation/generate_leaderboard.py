#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import statistics
from pathlib import Path

# ==========================================
# 目录配置
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
RES_DIR = BASE_DIR / "results"
OBJ_WER = RES_DIR / "objective" / "wer"
OBJ_SIM = RES_DIR / "objective" / "sim"
SUBJ_LOC = RES_DIR / "lalm" / "local"
SUBJ_GLO = RES_DIR / "lalm" / "global"
OUT_MD = BASE_DIR / "Leaderboard.md"

LANGS = ["en", "zh"]
C_DIMS = ["C1_Interaction_Naturalness", "C2_Global_Flow", "C3_Scene_Immersion"]
B_DIMS = ["B1_Instruction_Following", "B2_Prosody_Fluency", "B3_Local_Expressiveness"]
ALL_DIMS = B_DIMS + C_DIMS

DIM_LABELS = {
    "WER/CER": "WER/CER (↓)",
    "Spk_Sim": "Spk-Sim (↑)",
    "B1_Instruction_Following": "B1 指令遵循",
    "B2_Prosody_Fluency": "B2 韵律流畅",
    "B3_Local_Expressiveness": "B3 局部表现力",
    "C1_Interaction_Naturalness": "C1 交互自然度",
    "C2_Global_Flow": "C2 全局流畅",
    "C3_Scene_Immersion": "C3 场景沉浸"
}

def get_winner(val):
    if isinstance(val, dict):
        w = val.get("final_winner") or val.get("winner") or val.get("norm_winner") or "Tie"
    else:
        w = str(val)
    if w == "A": return "Model"
    if w == "B": return "GT"
    return w

def get_models():
    models = set()
    for d in [OBJ_WER, OBJ_SIM, SUBJ_LOC, SUBJ_GLO]:
        if d.exists():
            for m in d.iterdir():
                if m.is_dir(): models.add(m.name)
    return sorted(list(models))

def process_lang(lang):
    models = get_models()
    stats_all = {}
    
    for m in models:
        m_stats = {}
        
        # 1. 提取客观指标 WER/CER
        wer_file = OBJ_WER / m / lang / "results.jsonl"
        wers = []
        if wer_file.exists():
            for line in wer_file.read_text(encoding='utf-8').splitlines():
                if not line.strip(): continue
                d = json.loads(line)
                if d.get("wer_status") == "ok" and "wer" in d:
                    wers.append(float(d["wer"]))
        m_stats["WER/CER"] = f"{sum(wers)/len(wers):.4f}" if wers else "-"
        
        # 2. 提取客观指标 Spk-Sim
        sim_file = OBJ_SIM / m / lang / "main_sim.jsonl"
        sims = []
        if sim_file.exists():
            for line in sim_file.read_text(encoding='utf-8').splitlines():
                if not line.strip(): continue
                d = json.loads(line)
                if d.get("main_sim_status") == "ok" and d.get("main_sim_mean") is not None:
                    sims.append(float(d["main_sim_mean"]))
        m_stats["Spk_Sim"] = f"{sum(sims)/len(sims):.4f}" if sims else "-"
        
        # 3. 提取主观指标 (Global & Local)
        run_awrs = {d: [] for d in ALL_DIMS}
        
        # 读取 Global
        glo_dir = SUBJ_GLO / m / lang
        if glo_dir.exists():
            for r in [1, 2, 3]:
                f = glo_dir / f"run_{r}.jsonl"
                if f.exists():
                    w_t_n = {dim: [0,0,0] for dim in C_DIMS} # [win, tie, total]
                    for line in f.read_text(encoding='utf-8').splitlines():
                        if not line.strip(): continue
                        d = json.loads(line)
                        if d.get("status") == "ok":
                            res = d.get("result", {})
                            for dim in C_DIMS:
                                if dim in res:
                                    win = get_winner(res[dim])
                                    if win in ["Model", "Tie", "GT"]:
                                        w_t_n[dim][2] += 1
                                        if win == "Model": w_t_n[dim][0] += 1
                                        elif win == "Tie": w_t_n[dim][1] += 1
                    for dim in C_DIMS:
                        if w_t_n[dim][2] > 0:
                            run_awrs[dim].append((w_t_n[dim][0] + 0.5 * w_t_n[dim][1]) / w_t_n[dim][2])
                            
        # 读取 Local
        loc_dir = SUBJ_LOC / m / lang
        if loc_dir.exists():
            for r in [1, 2, 3]:
                f = loc_dir / f"run_{r}.jsonl"
                if f.exists():
                    w_t_n = {dim: [0,0,0] for dim in B_DIMS}
                    for line in f.read_text(encoding='utf-8').splitlines():
                        if not line.strip(): continue
                        d = json.loads(line)
                        if d.get("status") == "ok":
                            for u in d.get("result", {}).get("utterances", []):
                                for dim in B_DIMS:
                                    if dim in u:
                                        win = get_winner(u[dim])
                                        if win in ["Model", "Tie", "GT"]:
                                            w_t_n[dim][2] += 1
                                            if win == "Model": w_t_n[dim][0] += 1
                                            elif win == "Tie": w_t_n[dim][1] += 1
                    for dim in B_DIMS:
                        if w_t_n[dim][2] > 0:
                            run_awrs[dim].append((w_t_n[dim][0] + 0.5 * w_t_n[dim][1]) / w_t_n[dim][2])
        
        valid_subj = []
        for dim in ALL_DIMS:
            if run_awrs[dim]:
                mean_val = sum(run_awrs[dim]) / len(run_awrs[dim])
                std_val = statistics.stdev(run_awrs[dim]) if len(run_awrs[dim]) > 1 else 0.0
                m_stats[dim] = f"{mean_val*100:.1f}% &plusmn; {std_val*100:.1f}%"
                valid_subj.append(mean_val)
            else:
                m_stats[dim] = "-"
                
        if valid_subj:
            m_stats["Subj_Avg"] = sum(valid_subj) / len(valid_subj)
            m_stats["Subj_Avg_Str"] = f"**{m_stats['Subj_Avg']*100:.1f}%**"
        else:
            m_stats["Subj_Avg"] = -1
            m_stats["Subj_Avg_Str"] = "-"
            
        stats_all[m] = m_stats
        
    return stats_all

def generate_md_table(title, data):
    headers = ["模型 (Model)", DIM_LABELS["WER/CER"], DIM_LABELS["Spk_Sim"]] + [DIM_LABELS[d] for d in ALL_DIMS] + ["主观平均胜率"]
    md = f"### {title}\n\n| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    
    # 按照主观综合胜率降序排序
    sorted_data = sorted(data.items(), key=lambda x: x[1].get("Subj_Avg", -1), reverse=True)
    
    for m, row in sorted_data:
        cols = [m, row.get("WER/CER", "-"), row.get("Spk_Sim", "-")]
        for dim in ALL_DIMS:
            cols.append(row.get(dim, "-"))
        cols.append(row.get("Subj_Avg_Str", "-"))
        md += "| " + " | ".join(cols) + " |\n"
        
    return md + "\n<br>\n\n"

def main():
    print("==================================================")
    print(" 🔍 开始聚合 DRAMA-Eval 所有客观与主观评估结果...")
    print("==================================================")
    
    stats_en = process_lang("en")
    stats_zh = process_lang("zh")
    
    md_content = "# DRAMA-Eval Benchmark Leaderboard\n\n"
    md_content += "> **指标说明**：\n"
    md_content += "> * **WER/CER**: 词/字符错误率 (越低越好)\n"
    md_content += "> * **Spk-Sim**: 跨轮次角色音色余弦相似度 (越高越好)\n"
    md_content += "> * **主观AWR (对真人胜率)**: (Model胜局数 + 0.5 × Tie平局数) / 总有效局数 (3次盲测均值 ± 标准差)\n\n<hr>\n\n"
    
    md_content += generate_md_table("一、 英文 (EN) - 综合评估结果", stats_en)
    md_content += generate_md_table("二、 中文 (ZH) - 综合评估结果", stats_zh)
    
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(md_content.replace("&plusmn;", "±"))
    print(f"🎉 统计完成！详细结果表已生成于：{OUT_MD}")

if __name__ == "__main__":
    main()

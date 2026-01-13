#!/usr/bin/env python3
"""分析实验一的结果数据并生成汇总报告"""
import json
import os
import numpy as np
from collections import defaultdict
from datetime import datetime

def load_all_records():
    """加载所有实验记录（从recovered.json和jsonl）"""
    recovered_file = 'results/ijcai_experiments/results_recovered.json'
    jsonl_file = 'results/ijcai_experiments/results_main_comparison.jsonl'
    
    all_records = []
    seen = set()
    
    # 1. 从recovered.json读取
    if os.path.exists(recovered_file):
        with open(recovered_file, 'r') as f:
            data = json.load(f)
            results = data.get('results', {})
            for method, method_results in results.items():
                for budget_key, entries in method_results.items():
                    try:
                        budget = int(budget_key.split('_')[1])
                    except:
                        continue
                    for entry in entries:
                        try:
                            if isinstance(entry, dict):
                                seed = entry.get('seed')
                                final_score = entry.get('final_score', 0.0)
                                if seed is not None and final_score != 0.0:
                                    key = (method, budget, seed)
                                    if key not in seen:
                                        seen.add(key)
                                        all_records.append({
                                            'method': method,
                                            'budget': budget,
                                            'seed': seed,
                                            'final_score': final_score,
                                            'polarization_history': entry.get('polarization_history', []),
                                            'selected_nodes': entry.get('selected_nodes', []),
                                            'step_rewards': entry.get('step_rewards', [])
                                        })
                        except:
                            pass
    
    # 2. 从jsonl读取（去重）
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    method = rec.get('method')
                    budget = rec.get('budget')
                    seed = rec.get('seed')
                    final_score = rec.get('final_score', 0.0)
                    if method and budget is not None and seed is not None and final_score != 0.0:
                        key = (method, budget, seed)
                        if key not in seen:
                            seen.add(key)
                            all_records.append({
                                'method': method,
                                'budget': budget,
                                'seed': seed,
                                'final_score': final_score,
                                'polarization_history': rec.get('polarization_history', []),
                                'selected_nodes': rec.get('selected_nodes', []),
                                'step_rewards': rec.get('step_rewards', [])
                            })
                except:
                    pass
    
    return all_records

def generate_analysis():
    """生成完整的分析报告"""
    print("正在加载数据...")
    all_records = load_all_records()
    print(f"共加载 {len(all_records)} 条记录")
    
    methods = ['Random', 'High-Degree', 'PageRank', 'FINDER', 'Semantic-FINDER']
    budgets = [762, 1524, 3812]
    
    # 按方法和预算组织数据
    method_budget_data = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        method_budget_data[rec['method']][rec['budget']].append(rec['final_score'])
    
    # 生成JSON格式的详细分析
    analysis = {
        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_records': len(all_records),
        'method_comparison': {},
        'budget_analysis': {},
        'method_ranking': [],
        'relative_improvement': {},
        'budget_effect': {},
        'data_completeness': {}
    }
    
    # 1. 方法总体性能对比
    method_means = {}
    for method in methods:
        all_scores = []
        for budget in budgets:
            all_scores.extend(method_budget_data[method][budget])
        if all_scores:
            method_means[method] = np.mean(all_scores)
            analysis['method_comparison'][method] = {
                'mean': float(np.mean(all_scores)),
                'std': float(np.std(all_scores)),
                'min': float(np.min(all_scores)),
                'max': float(np.max(all_scores)),
                'median': float(np.median(all_scores)),
                'q25': float(np.percentile(all_scores, 25)),
                'q75': float(np.percentile(all_scores, 75)),
                'count': len(all_scores)
            }
    
    # 2. 方法排名
    sorted_methods = sorted(method_means.items(), key=lambda x: x[1])
    analysis['method_ranking'] = [
        {'rank': i+1, 'method': method, 'mean_polarization': float(mean)}
        for i, (method, mean) in enumerate(sorted_methods)
    ]
    
    # 3. 相对性能提升
    if sorted_methods:
        baseline_mean = sorted_methods[-1][1]  # 最差方法
        for method, mean in sorted_methods:
            improvement = (baseline_mean - mean) / baseline_mean * 100
            analysis['relative_improvement'][method] = float(improvement)
    
    # 4. 按预算分析
    for budget in budgets:
        budget_data = {}
        for method in methods:
            scores = method_budget_data[method][budget]
            if scores:
                budget_data[method] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'count': len(scores)
                }
        analysis['budget_analysis'][f'budget_{budget}'] = budget_data
    
    # 5. 预算效应
    for method in methods:
        budget_means = []
        for budget in budgets:
            scores = method_budget_data[method][budget]
            if scores:
                budget_means.append({'budget': budget, 'mean': float(np.mean(scores))})
        if budget_means:
            analysis['budget_effect'][method] = budget_means
    
    # 6. 数据完整性
    for method in methods:
        counts = {}
        total = 0
        for budget in budgets:
            count = len(method_budget_data[method][budget])
            counts[f'budget_{budget}'] = count
            total += count
        counts['total'] = total
        analysis['data_completeness'][method] = counts
    
    # 保存JSON分析结果
    json_output = 'results/ijcai_experiments/analysis_summary.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"JSON分析结果已保存到: {json_output}")
    
    # 生成Markdown报告
    report_lines = []
    report_lines.append("# 实验一：SOTA性能对比 - 数据分析报告\n")
    report_lines.append(f"**生成时间**: {analysis['analysis_time']}\n")
    report_lines.append(f"**总记录数**: {len(all_records)} 条\n")
    report_lines.append("---\n")
    
    # 1. 方法总体性能对比
    report_lines.append("## 1. 方法总体性能对比\n")
    report_lines.append("| 排名 | 方法 | 平均极化度 | 标准差 | 最小值 | 最大值 | 中位数 | 样本数 |")
    report_lines.append("|------|------|------------|--------|--------|--------|--------|--------|")
    
    for rank_info in analysis['method_ranking']:
        rank = rank_info['rank']
        method = rank_info['method']
        comp = analysis['method_comparison'][method]
        report_lines.append(
            f"| {rank} | {method} | {comp['mean']:.8f} | {comp['std']:.8f} | "
            f"{comp['min']:.8f} | {comp['max']:.8f} | {comp['median']:.8f} | {comp['count']} |"
        )
    report_lines.append("")
    
    # 2. 相对性能提升
    report_lines.append("## 2. 相对性能提升（相对于最差方法）\n")
    report_lines.append("| 方法 | 平均极化度 | 相对提升 |")
    report_lines.append("|------|------------|----------|")
    for rank_info in analysis['method_ranking']:
        method = rank_info['method']
        mean = rank_info['mean_polarization']
        improvement = analysis['relative_improvement'][method]
        report_lines.append(f"| {method} | {mean:.8f} | {improvement:.2f}% |")
    report_lines.append("")
    
    # 3. 不同预算下的性能
    report_lines.append("## 3. 不同预算下的性能分析\n")
    for budget in budgets:
        report_lines.append(f"### Budget = {budget}\n")
        report_lines.append("| 方法 | 平均极化度 | 标准差 | 样本数 |")
        report_lines.append("|------|------------|--------|--------|")
        budget_key = f'budget_{budget}'
        budget_data = analysis['budget_analysis'][budget_key]
        
        # 按平均极化度排序
        sorted_budget = sorted(budget_data.items(), key=lambda x: x[1]['mean'])
        for method, stats in sorted_budget:
            report_lines.append(f"| {method} | {stats['mean']:.8f} | {stats['std']:.8f} | {stats['count']} |")
        report_lines.append("")
        
        # 排名
        methods_ranked = [m for m, _ in sorted_budget]
        report_lines.append(f"**Budget {budget} 排名**: {' > '.join(methods_ranked)} (极化度越低越好)")
        report_lines.append("")
    
    # 4. 预算效应分析
    report_lines.append("## 4. 预算效应分析（预算增加对性能的影响）\n")
    report_lines.append("| 方法 | Budget 762 | Budget 1524 | Budget 3812 | 趋势 |")
    report_lines.append("|------|------------|-------------|-------------|------|")
    for method in methods:
        means = []
        for budget in budgets:
            budget_key = f'budget_{budget}'
            if method in analysis['budget_analysis'][budget_key]:
                means.append(f"{analysis['budget_analysis'][budget_key][method]['mean']:.8f}")
            else:
                means.append("N/A")
        
        # 判断趋势
        valid_means = [float(m) for m in means if m != "N/A"]
        if len(valid_means) >= 2:
            if valid_means[-1] < valid_means[0]:
                trend = "📉 改善"
            elif valid_means[-1] > valid_means[0]:
                trend = "📈 恶化"
            else:
                trend = "➡️ 稳定"
        else:
            trend = "N/A"
        
        report_lines.append(f"| {method} | {' | '.join(means)} | {trend} |")
    report_lines.append("")
    
    # 5. 数据完整性检查
    report_lines.append("## 5. 数据完整性检查\n")
    report_lines.append("| 方法 | Budget 762 | Budget 1524 | Budget 3812 | 总计 |")
    report_lines.append("|------|------------|-------------|-------------|------|")
    for method in methods:
        completeness = analysis['data_completeness'][method]
        report_lines.append(
            f"| {method} | {completeness['budget_762']} | {completeness['budget_1524']} | "
            f"{completeness['budget_3812']} | {completeness['total']} |"
        )
    report_lines.append("")
    report_lines.append(f"**总任务数**: 750 (5方法 × 3预算 × 50seed)")
    report_lines.append(f"**已完成**: {len(all_records)}")
    report_lines.append(f"**完成率**: {len(all_records)/750*100:.1f}%")
    report_lines.append("")
    
    # 6. 关键发现
    report_lines.append("## 6. 关键发现\n")
    best_method = analysis['method_ranking'][0]['method']
    worst_method = analysis['method_ranking'][-1]['method']
    best_score = analysis['method_ranking'][0]['mean_polarization']
    worst_score = analysis['method_ranking'][-1]['mean_polarization']
    improvement = analysis['relative_improvement'][best_method]
    
    report_lines.append(f"- **最佳方法**: {best_method} (平均极化度: {best_score:.8f})")
    report_lines.append(f"- **最差方法**: {worst_method} (平均极化度: {worst_score:.8f})")
    report_lines.append(f"- **性能提升**: {best_method} 相比 {worst_method} 提升了 {improvement:.2f}%")
    report_lines.append("")
    
    # 保存Markdown报告
    md_output = 'results/ijcai_experiments/analysis_report.md'
    with open(md_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Markdown报告已保存到: {md_output}")
    
    print("\n✅ 分析完成！")

if __name__ == '__main__':
    generate_analysis()

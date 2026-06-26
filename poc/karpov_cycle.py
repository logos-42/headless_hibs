"""
Twistor-LMT 卡帕斯循环系统 (Karpov Cycle)
==========================================
基于 Ollama 本地 LLM 的自动化迭代改进系统

核心流程:
1. 分析当前代码
2. LLM 生成改进建议
3. 自动应用改进
4. 测试验证
5. 循环迭代

使用方式:
    python karpov_cycle.py --iterations 5
    python karpov_cycle.py --model llama2 --iterations 3
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import requests


# ============================================================================
# Ollama API 客户端
# ============================================================================

class OllamaClient:
    """Ollama LLM 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self._check_connection()
    
    def _check_connection(self):
        """检查 Ollama 连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✓ 已连接到 Ollama")
                print(f"  可用模型：{[m['name'] for m in models]}")
            else:
                print(f"⚠ Ollama 返回状态码：{response.status_code}")
        except requests.exceptions.ConnectionError:
            print("✗ 无法连接到 Ollama，请确保 Ollama 正在运行")
            print("  启动命令：ollama serve")
            raise
        except Exception as e:
            print(f"⚠ 连接 Ollama 时出错：{e}")
            raise
    
    def chat(self, prompt: str, system: str = "", stream: bool = False) -> str:
        """发送聊天请求"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result['message']['content']
        except Exception as e:
            print(f"✗ Ollama API 调用失败：{e}")
            return ""
    
    def generate(self, prompt: str, system: str = "") -> str:
        """发送生成请求"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result['response']
        except Exception as e:
            print(f"✗ Ollama API 调用失败：{e}")
            return ""


# ============================================================================
# 代码分析器
# ============================================================================

class CodeAnalyzer:
    """代码分析器"""
    
    def __init__(self, repo_path: str = '.'):
        self.repo_path = Path(repo_path)
        self.python_files = list(self.repo_path.glob('**/*.py'))
    
    def get_file_stats(self) -> Dict:
        """获取文件统计信息"""
        stats = {
            'total_files': len(self.python_files),
            'total_lines': 0,
            'files': []
        }
        
        for file in self.python_files:
            if 'test_' not in str(file) and '__pycache__' not in str(file):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        stats['total_lines'] += line_count
                        stats['files'].append({
                            'path': str(file.relative_to(self.repo_path)),
                            'lines': line_count,
                        })
                except:
                    pass
        
        return stats
    
    def get_recent_changes(self, last_commit: str = 'HEAD~1') -> str:
        """获取最近的代码变更"""
        try:
            result = subprocess.run(
                ['git', 'diff', last_commit, '--', '*.py'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except:
            return ""
    
    def analyze_code_quality(self) -> Dict:
        """分析代码质量"""
        issues = []
        
        for file in self.python_files:
            if 'test_' in str(file) or '__pycache__' in str(file):
                continue
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # 检查项
                if len(lines) > 500:
                    issues.append({
                        'file': str(file.relative_to(self.repo_path)),
                        'type': 'long_file',
                        'message': f'文件过长 ({len(lines)} 行)',
                        'severity': 'low'
                    })
                
                if 'TODO' in content or 'FIXME' in content:
                    issues.append({
                        'file': str(file.relative_to(self.repo_path)),
                        'type': 'todo',
                        'message': '存在 TODO/FIXME 标记',
                        'severity': 'info'
                    })
                
                # 检查函数长度
                current_func = None
                func_lines = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('def '):
                        if func_lines > 100:
                            issues.append({
                                'file': str(file.relative_to(self.repo_path)),
                                'type': 'long_function',
                                'message': f'函数过长 ({func_lines} 行): {current_func}',
                                'severity': 'medium',
                                'line': i
                            })
                        current_func = line.strip()
                        func_lines = 0
                    func_lines += 1
                
            except Exception as e:
                pass
        
        return {
            'issues': issues,
            'total_issues': len(issues)
        }
    
    def get_code_context(self, max_files: int = 5) -> str:
        """获取代码上下文 (用于 LLM 分析)"""
        context = []
        
        # 选择关键文件
        key_files = [
            'twistor_LMT.py',
            'twistor_100m_config.py',
            'auto_evolution.py',
            'auto_archive.py',
        ]
        
        for filename in key_files:
            file_path = self.repo_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 只取前 500 行
                        lines = content.split('\n')[:500]
                        context.append(f"=== {filename} ===\n" + '\n'.join(lines))
                except:
                    pass
        
        return '\n\n'.join(context)


# ============================================================================
# 测试验证器
# ============================================================================

class TestValidator:
    """测试验证器"""
    
    def __init__(self, repo_path: str = '.'):
        self.repo_path = Path(repo_path)
    
    def run_tests(self) -> Tuple[bool, str]:
        """运行测试"""
        print("运行测试...")
        
        # 测试 1: 导入测试
        print("  测试 1: 导入测试...")
        try:
            import torch
            from poc.twistor_100m_config import StackedTwistorLMT, TwistorLMTConfig
            
            config = TwistorLMTConfig.small()
            model = StackedTwistorLMT(config)
            
            x = torch.randn(10, 1, 1)
            with torch.no_grad():
                y = model(x)
            
            assert y.shape == (10, 1, 1)
            print("    ✓ 导入测试通过")
        except Exception as e:
            print(f"    ✗ 导入测试失败：{e}")
            return False, f"Import test failed: {e}"
        
        # 测试 2: 进化系统测试
        print("  测试 2: 进化系统测试...")
        try:
            from poc.auto_evolution import ArchitectureConfig
            
            config = ArchitectureConfig(hidden_dim=64, n_layers=2)
            assert config.hidden_dim == 64
            assert config.n_layers == 2
            print("    ✓ 进化系统测试通过")
        except Exception as e:
            print(f"    ✗ 进化系统测试失败：{e}")
            return False, f"Evolution test failed: {e}"
        
        # 测试 3: 存档系统测试
        print("  测试 3: 存档系统测试...")
        try:
            from poc.auto_archive import GitArchiveSystem
            
            archive = GitArchiveSystem()
            has_changes, branch = archive.check_git_status()
            print("    ✓ 存档系统测试通过")
        except Exception as e:
            print(f"    ✗ 存档系统测试失败：{e}")
            return False, f"Archive test failed: {e}"
        
        print("\n✓ 所有测试通过")
        return True, "All tests passed"


# ============================================================================
# 卡帕斯循环系统
# ============================================================================

class KarpovCycle:
    """
    卡帕斯循环系统
    
    自动化迭代改进流程:
    1. 分析当前代码
    2. LLM 生成改进建议
    3. 自动应用改进
    4. 测试验证
    5. 循环迭代
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        repo_path: str = '.',
        auto_apply: bool = False,
    ):
        self.ollama = ollama_client
        self.analyzer = CodeAnalyzer(repo_path)
        self.validator = TestValidator(repo_path)
        self.repo_path = Path(repo_path)
        self.auto_apply = auto_apply
        
        self.history = []
    
    def run_cycle(self, iteration: int) -> Dict:
        """运行一次卡帕斯循环"""
        print(f"\n{'='*70}")
        print(f"卡帕斯循环 - 迭代 {iteration}")
        print(f"{'='*70}")
        
        cycle_result = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        # 步骤 1: 分析代码
        print("\n步骤 1: 分析代码")
        print("-" * 40)
        
        file_stats = self.analyzer.get_file_stats()
        code_quality = self.analyzer.analyze_code_quality()
        code_context = self.analyzer.get_code_context()
        
        print(f"  文件数：{file_stats['total_files']}")
        print(f"  总行数：{file_stats['total_lines']}")
        print(f"  代码问题：{code_quality['total_issues']}")
        
        cycle_result['steps']['analysis'] = {
            'file_stats': file_stats,
            'code_quality': code_quality,
        }
        
        # 步骤 2: LLM 生成改进建议
        print("\n步骤 2: LLM 生成改进建议")
        print("-" * 40)
        
        system_prompt = """你是一个专业的 Python 代码审查专家。
你的任务是分析代码并提供具体的改进建议。
请提供：
1. 代码质量评估
2. 3-5 个具体改进建议
3. 优先级排序
4. 如果可能，提供代码示例"""

        analysis_prompt = f"""请分析以下 Twistor-LMT 项目代码:

文件统计:
- 总文件数：{file_stats['total_files']}
- 总行数：{file_stats['total_lines']}

代码问题:
{json.dumps(code_quality['issues'][:5], indent=2, ensure_ascii=False)}

代码上下文:
{code_context[:5000]}

请提供具体的改进建议。"""

        suggestions = self.ollama.chat(analysis_prompt, system_prompt)
        
        print(f"  LLM 建议:\n{suggestions[:500]}...")
        
        cycle_result['steps']['suggestions'] = suggestions
        
        # 步骤 3: 生成改进代码
        print("\n步骤 3: 生成改进代码")
        print("-" * 40)
        
        if self.auto_apply:
            code_prompt = f"""基于以下改进建议，生成具体的代码改进:

{suggestions}

请提供:
1. 需要修改的文件名
2. 修改前的代码
3. 修改后的代码
4. 使用 diff 格式

格式:
```diff
--- a/filename.py
+++ b/filename.py
@@ -1,5 +1,6 @@
 ...
```
"""
            
            code_changes = self.ollama.chat(code_prompt, system_prompt)
            print(f"  生成的代码改进:\n{code_changes[:500]}...")
            
            cycle_result['steps']['code_changes'] = code_changes
            
            # 步骤 4: 应用改进 (如果启用)
            print("\n步骤 4: 应用改进")
            print("-" * 40)
            print("  ⚠️ 自动应用功能暂未实现，请手动应用")
            
            cycle_result['steps']['applied'] = False
        
        # 步骤 5: 测试验证
        print("\n步骤 5: 测试验证")
        print("-" * 40)
        
        test_passed, test_message = self.validator.run_tests()
        
        print(f"  测试结果：{'✓ 通过' if test_passed else '✗ 失败'}")
        print(f"  {test_message}")
        
        cycle_result['steps']['test'] = {
            'passed': test_passed,
            'message': test_message
        }
        
        # 保存历史
        self.history.append(cycle_result)
        
        return cycle_result
    
    def run_iterations(self, n_iterations: int = 5):
        """运行多次迭代"""
        print("="*70)
        print("Twistor-LMT 卡帕斯循环系统")
        print("="*70)
        print(f"迭代次数：{n_iterations}")
        print(f"自动应用：{self.auto_apply}")
        print(f"LLM 模型：{self.ollama.model}")
        
        for i in range(1, n_iterations + 1):
            result = self.run_cycle(i)
            
            # 保存迭代历史
            self._save_history()
            
            # 如果测试失败，询问是否继续
            if not result['steps']['test']['passed']:
                print("\n⚠️ 测试失败！")
                cont = input("是否继续下一次迭代？(y/n): ")
                if cont.lower() != 'y':
                    print("已终止迭代")
                    break
        
        # 最终总结
        print("\n" + "="*70)
        print("卡帕斯循环完成")
        print("="*70)
        
        passed_iterations = sum(
            1 for r in self.history 
            if r['steps']['test']['passed']
        )
        
        print(f"总迭代次数：{n_iterations}")
        print(f"成功迭代：{passed_iterations}")
        print(f"失败迭代：{n_iterations - passed_iterations}")
        print(f"成功率：{passed_iterations/n_iterations*100:.1f}%")
        
        # 保存最终报告
        self._save_final_report()
    
    def _save_history(self):
        """保存迭代历史"""
        history_file = self.repo_path / 'karpov_history.json'
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def _save_final_report(self):
        """保存最终报告"""
        report_file = self.repo_path / 'karpov_report.md'
        
        report = f"""# 卡帕斯循环最终报告

**生成时间**: {datetime.now().isoformat()}
**总迭代次数**: {len(self.history)}

## 迭代历史

"""
        
        for i, result in enumerate(self.history, 1):
            report += f"""
### 迭代 {i}

**时间**: {result['timestamp']}
**测试结果**: {'✓ 通过' if result['steps']['test']['passed'] else '✗ 失败'}

**改进建议**:
{result['steps'].get('suggestions', '无')[:500]}

---

"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n最终报告已保存到：{report_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Twistor-LMT 卡帕斯循环系统')
    parser.add_argument('-i', '--iterations', type=int, default=3,
                       help='迭代次数 (默认：3)')
    parser.add_argument('-m', '--model', type=str, default='llama2',
                       help='Ollama 模型名称 (默认：llama2)')
    parser.add_argument('-u', '--url', type=str, default='http://localhost:11434',
                       help='Ollama API 地址 (默认：http://localhost:11434)')
    parser.add_argument('-a', '--auto-apply', action='store_true',
                       help='自动应用改进 (实验性)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='仅分析，不迭代')
    
    args = parser.parse_args()
    
    # 创建客户端
    print("正在连接 Ollama...")
    try:
        ollama = OllamaClient(base_url=args.url, model=args.model)
    except Exception as e:
        print(f"\n✗ 无法连接 Ollama: {e}")
        print("\n请确保:")
        print("  1. Ollama 已安装")
        print("  2. 运行 'ollama serve'")
        print("  3. 模型已下载：'ollama pull llama2'")
        return
    
    # 创建分析器
    analyzer = CodeAnalyzer()
    
    if args.analyze_only:
        # 仅分析模式
        print("\n" + "="*70)
        print("代码分析报告")
        print("="*70)
        
        stats = analyzer.get_file_stats()
        quality = analyzer.analyze_code_quality()
        
        print(f"\n文件统计:")
        print(f"  总文件数：{stats['total_files']}")
        print(f"  总行数：{stats['total_lines']}")
        
        print(f"\n代码问题:")
        for issue in quality['issues'][:10]:
            print(f"  - [{issue['severity']}] {issue['file']}: {issue['message']}")
        
        return
    
    # 创建卡帕斯循环系统
    karpov = KarpovCycle(
        ollama_client=ollama,
        auto_apply=args.auto_apply,
    )
    
    # 运行迭代
    karpov.run_iterations(n_iterations=args.iterations)


if __name__ == "__main__":
    main()

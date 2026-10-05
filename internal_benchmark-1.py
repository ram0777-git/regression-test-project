#!/usr/bin/env python3
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

TASKS = [
("math", "mathlib.py", '''def clamp(x, lo, hi):
    if lo > hi: raise ValueError("lo must be <= hi")
    return max(lo, min(x, hi))

def average(values):
    values=list(values)
    if not values: raise ValueError("empty")
    return sum(values)/len(values)

def safe_divide(a,b):
    if b==0: return None
    return a/b
''', '''Cover mathlib.py: clamp, average, and safe_divide. In scope: normal values,
boundaries, empty input, and division by zero. Add tests only under tests/.
Do not assert private implementation details.''', [
("clamp", "return max(lo, min(x, hi))", "return min(lo, max(x, hi))"),
("clamp_error", 'raise ValueError("lo must be <= hi")', 'raise ValueError("bad range")'),
("average", "return sum(values)/len(values)", "return sum(values)/(len(values)+1)"),
("empty", 'raise ValueError("empty")', 'return 0'),
("zero", "if b==0: return None", "if b==0: return 0"),
]),
("account", "account.py", '''class Account:
    def __init__(self,balance=0): self.balance=balance
    def deposit(self,amount):
        if amount<=0: raise ValueError("amount must be positive")
        self.balance += amount
        return self.balance
    def withdraw(self,amount):
        if amount<=0: raise ValueError("amount must be positive")
        if amount>self.balance: raise ValueError("insufficient funds")
        self.balance -= amount
        return self.balance
    def transfer(self,other,amount):
        self.withdraw(amount); other.deposit(amount); return self.balance

def discount(price,percent): return price*(1-percent/100)
''', '''Cover account.py: Account.deposit, Account.withdraw, and Account.transfer.
In scope: positive amounts, insufficient funds, transfers, and balances.
Out of scope: discount. Add tests only under tests/.''', [
("deposit", "self.balance += amount", "self.balance -= amount"),
("deposit_error", 'raise ValueError("amount must be positive")', 'raise ValueError("deposit invalid")'),
("withdraw", "self.balance -= amount", "self.balance += amount"),
("withdraw_limit", "if amount>self.balance:", "if amount>=self.balance:"),
("transfer", "other.deposit(amount)", "other.deposit(amount+1)"),
]),
("state", "state_lib.py", '''def parse_level(value):
    value=str(value).lower()
    if value in ("low","medium","high"): return value
    raise ValueError("invalid level")
class Counter:
    def __init__(self): self.value=0
    def increment(self,amount=1): self.value += amount; return self.value
    def reset(self): self.value=0; return self.value
def toggle(flag): return not flag
''', '''Cover state_lib.py: parse_level, Counter.increment, Counter.reset, and toggle.
In scope: valid/invalid levels, counter state changes, reset, and toggling.
Add tests only under tests/.''', [
("level", 'if value in ("low","medium","high"):', 'if value in ("low","medium"):'),
("level_error", 'raise ValueError("invalid level")', 'return "unknown"'),
("increment", "self.value += amount", "self.value -= amount"),
("reset", "self.value=0; return self.value", "self.value=1; return self.value"),
("toggle", "return not flag", "return flag"),
]),
]

def sh(cmd, cwd, timeout=120):
    return subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)

def setup(root, task):
    name, mod, source, statement, bugs = task
    (root/mod).write_text(source); (root/'task.txt').write_text(statement); (root/'tests').mkdir()
    sh(['git','init','-q'],root); sh(['git','config','user.email','bench@example.com'],root); sh(['git','config','user.name','Benchmark'],root)
    sh(['git','add','.'],root); sh(['git','commit','-qm','baseline'],root)

def pytest(root):
    r=sh([sys.executable,'-m','pytest','-q','tests'],root,60)
    return r.returncode==0,r.stdout[-1500:]

def evaluate(agent, task, timeout):
    with tempfile.TemporaryDirectory(prefix='rtbench_') as td:
        root=Path(td); setup(root,task)
        r=sh([sys.executable,str(Path(agent).resolve()),str(root),'--task',str(root/'task.txt'),
              '--max-rounds','3','--max-mutants','40','--timeout',str(timeout),'--no-llm'],Path.cwd(),timeout+30)
        ok,out=pytest(root)
        tests=list((root/'tests').glob('*.py'))
        result=[]; caught=0
        if not ok or not tests:
            return 0,len(task[4]),[],out
        for bug,old,new in task[4]:
            sh(['git','reset','--hard','-q','HEAD'],root)
            p=root/task[1]; s=p.read_text()
            if old not in s:
                result.append([bug,False]); continue
            p.write_text(s.replace(old,new,1))
            bad,_=pytest(root); hit=not bad; caught+=hit; result.append([bug,hit])
        return caught,len(task[4]),result,''

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--agent',required=True); ap.add_argument('--timeout',type=int,default=180); a=ap.parse_args()
    total=caught=0; report=[]
    print('LOCAL REGRESSION BENCHMARK (reference only)')
    print('='*55)
    for task in TASKS:
        try: c,t,b,e=evaluate(a.agent,task,a.timeout)
        except Exception as ex: c,t,b,e=0,len(task[4]),[],repr(ex)
        caught+=c; total+=t; report.append({'task':task[0],'caught':c,'total':t,'bugs':b,'error':e})
        print(f'{task[0]}: {c}/{t} caught')
    score=100*caught/total if total else 0
    data={'official':False,'score_percent':round(score,2),'caught':caught,'total':total,'tasks':report}
    Path('internal_benchmark_report.json').write_text(json.dumps(data,indent=2))
    print('='*55); print(f'FINAL SCORE: {score:.2f}% ({caught}/{total})')
    print('Report: internal_benchmark_report.json')
    print('This is NOT the official benchmark score.')
if __name__=='__main__': main()

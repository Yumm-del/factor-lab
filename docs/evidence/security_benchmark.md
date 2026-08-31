# DSL 安全基准

该基准完全离线运行，不调用 LLM API，不读取行情或用户文件。

- 总用例：69
- 通过：69/69
- 合法公式通过率：100.0%
- 非法或危险输入拦截率：100.0%

| 类别 | 用例数 | 通过数 |
|---|---:|---:|
| injection | 10 | 10 |
| invalid_parameter | 10 | 10 |
| invalid_structure | 15 | 15 |
| malformed_json | 10 | 10 |
| resource_limit | 2 | 2 |
| unknown_operator | 10 | 10 |
| valid | 12 | 12 |

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\security_benchmark.py
```

完整逐例结果见 `security_benchmark.json`。

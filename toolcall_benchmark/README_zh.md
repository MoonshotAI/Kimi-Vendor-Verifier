# toolcall_benchmark JSONL 数据说明

本目录存放用于工具调用（tool call）能力评测的样本数据，采用 **JSONL（JSON Lines）** 格式：

- 每行是一个独立的 JSON 对象，代表一次完整的模型对话样本。
- 每行内部结构相同，可直接用 `json.loads(line)` 逐行解析。
- 本目录仅提供 thinking benchmark：后续模型没有 non-thinking 需求，因此没有 non-thinking 的 toolcall 数据集（原 `toolcall_non_thinking_samples.cleaned.jsonl` 已删除）。

> 注：本数据集已完成全量去标识化，可对外共享。凭据/令牌已替换为类型化占位符
> （如 `[REDACTED_SECRET]`）；手机号、邮箱、地址、IP、平台 ID、路径用户名等
> 隐私信息已替换为一致的合成或掩码值（如 `188****2656`、`uNNNN@example.com`，[TELEGRAM_ID]，
> `user9727`）。人名脱敏使用小说人物命名替换脱敏，见下文。
>
> **标识符替换的安全性说明。** 平台 ID（飞书、Discord
> snowflake、Telegram 数字 ID等）的标识熵几乎均匀分布在每一个字符上：只掩码前缀或
> 后缀会残留一个全局唯一、跨数据集稳定不变的假名标识，仍可在公开日志中检索、
> 与其他数据源关联，即数据主体依然可被重识别。所以这类标识符已替换值为 HMAC 派生
> 的合成值：密钥仅保存在本地、不随数据集发布，
> 任何持有者都无法从合成值反推或关联到原始标识，发布的数据集与源身份之间是
> 构造上不可关联的（unlinkable）。合成值保持原格式仅为满足下游的格式校验，其
> 本身不携带任何关于数据主体的信息。

## 每行顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 样本唯一标识（UUID）。 |
| `created_at` | `str` | 样本创建时间（带时区），例如 `2026-04-26 11:59:21+08:00`。 |
| `prompt_tokens` | `int` | 输入 prompt 的 token 数。 |
| `completion_tokens` | `int` | 模型输出的 token 数。 |
| `cached_tokens` | `int` | 命中缓存的 token 数。 |
| `total_tokens` | `int` | 总 token 数（约等于 `prompt_tokens + completion_tokens`）。 |
| `msg_count` | `int` | 对话总消息数。 |
| `_repeat_run` | `dict` | 重复/循环调用分析结果，见下文。 |
| `tools` | `list` | 本次对话中可用的工具列表，每个元素为 OpenAI 风格的 function tool 定义。 |
| `thinking` | `dict` | 思考模式配置，仅包含 `type` 字段：`enabled` 或 `disabled`。 |
| `messages` | `list` | 完整对话历史，每个元素是一条消息对象，见下文。 |

## `_repeat_run` 字段

用于标注该样本是否出现“重复调用同一工具”的异常模式，本数据来源于线上报错数据，因此通过此字段记录错误位置和内容，在kimi_toolcall中会按照下面记录的索引使用数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `tool_name` | `str` | 被判定为重复调用的工具名。 |
| `run_first_idx` | `int` | 工具首次正常调用在 `messages` 中的索引。 |
| `repeat_start_idx` | `int` | 重复调用段的起始消息索引。 |
| `run_last_idx` | `int` | 重复调用段的结束消息索引。 |
| `run_length` | `int` | 重复调用段包含的消息数量。 |
| `loop_phrase_hits` | `int` | 检测到循环提示语的次数。 |
| `error_phrase_hits` | `int` | 检测到错误提示语的次数。 |
| `tool_result_error_count` | `int` | 工具返回 error 的次数。 |
| `tool_result_total` | `int` | 该工具被调用的总次数。 |
| `tool_result_error_ratio` | `float` | 工具返回 error 的比例。 |
| `sample_error_excerpts` | `list` | 典型错误片段示例，每个元素包含 `msg_idx` 和 `excerpt`。 |

## `tools` 字段

标准 OpenAI function tool 数组，每个元素格式如下：

```json
{
  "type": "function",
  "function": {
    "name": "read",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": { ... },
      "required": ["path"]
    }
  }
}
```

## `messages` 字段

按时间顺序排列的对话消息数组。常见消息角色及字段：

| `role` | 常见字段 | 说明 |
|--------|----------|------|
| `user` | `role`, `content` | 用户输入。 |
| `assistant` | `role`, `content`, `tool_calls`, `reasoning_content` | 模型回复。`tool_calls` 为模型发起的工具调用列表；`reasoning_content` 为 thinking 模型输出的推理内容（非 thinking 样本中通常不存在）。 |
| `tool` | `role`, `content`, `tool_call_id` | 工具执行结果，`tool_call_id` 对应 `assistant.tool_calls[].id`。 |

`assistant.tool_calls` 数组中每个元素包含：

| 字段 | 说明 |
|------|------|
| `id` | 工具调用 ID，供 `tool` 消息回传。 |
| `type` | 固定为 `function`。 |
| `function` | 包含 `name`（工具名）和 `arguments`（JSON 字符串形式的参数）。 |

## 使用示例

```python
import json

with open("toolcall_benchmark/toolcall_thinking_samples.stable2.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        print(sample["id"], sample["thinking"])
        for msg in sample["messages"][:5]:
            print(" ", msg["role"], msg.keys())
```

## 人名脱敏（小说人物命名）
`toolcall_thinking_samples.stable2.jsonl`（70 条）中, 除新闻内容中的公众人物名未替换，其他所有人名已统一替换为小说人物名：中文名用《水浒传》人物，英文名用《战争与和平》人物。覆盖真实人名（团队模板、博主名、教材署名）。
规则：
- 同一人 = 同一小说身份，同一人的所有形态（全名/名/姓/别名/大小写变体）映射到同一小说人物。公众人物在同一样本内的中英文形态统一为一个《战争与和平》名；纯中文语境样本则用《水浒传》名。
- 同一小说名在不同样本中可指代不同真实人物（样本相互独立）。
- 按样本限定范围替换；拉丁名使用词边界匹配；掩码形态（`X*`）排除 Markdown 粗体邻接。
例外（保留原样）：
- 合成 ID：`user8402`、`user3617`、`user4547`、`user0041`, `user9727`。


此处仅列出每个样本使用的小说人物名：

| 样本 | 使用的小说人物名（含地名） |
|---|---|
| 07e927dc | 宋江、卢俊义、吴用、公孙胜、关胜、林冲、秦明、呼延灼、Pierre Bezukhov |
| 0fbec37d | Speranski、Langeron、Alexander Langeron、Dokhturov、Berthier、Bessieres、Oudinot、Rapp、Savary、Friant、Pierre、Nikolai Rostov、Platon Karataev、Balashev、Murat、Bagration |
| 103404ca | 郁保四、鲁智深、Lu Zhishen、ZhiShen Lu、Sergei Wolzogen、A. Speranski、Vasily Balashev、Z. Zherkov、M. Willarski、Ilya Ermolov、Daniel Oudinot、Heming Bilibin、Haibo Tushin、Nikolai Rostov、Andrei Bolkonsky、Pierre Bezukhov、Boris Drubetskoy、Anatole Kuragin、Fedor Dolokhov、Platon Karataev、Hippolyte Kuragin、Kuzma Timokhin、zhishen-lu、davout、berthier |
| 15c998ea | 白胜、段景住、高俅、蔡京 |
| 24818454 | Dmitri Rostov、Kirill Rostov、Anton Karagin、Roman Karagin、Semyon Kuragin、Taras Dolokhov、Prokhor Denisov、Efim Tushin、Kuzma Timokhin、Gavrilo Berg、Pavel Bilibin、Stepan Nesvitsky、YURI ZHERKOV、Lev Raevsky、Adrian Pfuel、Matvey Weyrother、Osip Langeron、Innokenty Dokhturov、Foma Ermolov、Hariton Miloradovich、Akim Murat、Viktor Davout、Platon Ney、Zakhar Berthier、Naum Caulaincourt、Lazar Bessieres、Orest Oudinot、Karp Rapp、Fadei Savary、Uvar Friant、Vlas Mortemart、Yefim Willarski、Anatole、Andrei Bolkonsky、Pelageya、Pierre Bezukhov |
| 2560e391 | 索超、戴宗、刘唐、李逵、史进、穆弘、雷横、李俊、阮小二、张横、阮小五、张顺、阮小七、杨雄、石秀、解珍、解宝、燕青、朱武、黄信、孙立、宣赞、郝思文、韩滔、彭玘、单廷珪、魏定国、萧让、裴宣、欧鹏、邓飞、燕顺、杨林、凌振、蒋敬、吕方、郭盛、安道全、皇甫端、王英、扈三娘、鲍旭、樊瑞、孔明、孔亮、项充、李衮、金大坚、朱贵、朱富、蔡福、蔡庆、李立、李云、焦挺、石勇、孙新、张青、晁盖、林冲、武松、鲁智深、时迁、Berthier、Bessieres、Mortier、高俅 |
| 33328b30 | Karataev |
| 35f5ee72 | 花荣 |
| 36958f65 | Julie |
| 385a2538 | Andrei Bolkonsky、Andrei |
| 39b732e9 | Platov |
| 400383aa | 秦明、Anna Pavlovna |
| 42f7bd41 | 花荣、马麟、扈三娘、张清、顾大嫂、安道全、李俊 |
| 4af405be | 柴进、高俅 |
| 5c549feb | 柴进、高俅 |
| 5e5db8b1 | 李应 |
| 6e95bf06 | 朱仝、鲁智深、武松、董平、张清、杨志、王定六 |
| 6f9d5dd3 | 花荣、马麟、扈三娘、张清、顾大嫂、安道全、李俊 |
| 76fc47c5 | Dolokhov |
| 83a8aa02 | 徐宁、白胜、蔡京、Rostopchin、Lavrushka、Alpatych、Ilyin |
| 904336b5 | 吴用、朱武、柴进、宣赞、扈三娘、郭盛、林冲、秦明、呼延灼、关胜、徐宁、索超、戴宗、卢俊义、Wu Yong |
| 922c7d05 | 蔡京、李逵、燕青、武松 |
| 97f46ffc | 花荣、童威、童猛、孟康、侯健、陈达、杨春、郑天寿、陶宗旺、宋清、乐和、龚旺、丁得孙、穆春、曹正、宋万、杜迁、江州、沧州、Denisov、Hippolyte、安道全、索超、欧鹏、邓飞、顾大嫂、孙二娘、李师师、武松、林冲行记、鲁智深周末去哪儿、扈三娘、石秀、二龙山故事多、燕青周边游 |
| 98b2ac59 | 花荣 |
| a58a42ff | 薛永、施恩、朱仝、杨志、史进、邹润、Ilya Rostov、Compans、Gudin、段景住、库图佐夫、索尼娅、鲍里斯、多洛霍夫、宣赞 |
| a61d18d1 | 时迁、杨林、史文恭、王英、Bezukhov、扈三娘、孙二娘、李师师、顾大嫂 |
| a6d1edb7 | 卢俊义 |
| a741e205 | 花荣、Bolkonsky、Rostov |
| b5029d9e | 花荣、李忠 |
| b5e03ba7 | 周通、汤隆 |
| c221feb3 | 花荣、杜兴、邹渊、邹润、宋江、扈三娘、关胜、鲁智深、孙二娘、秦明、白秀英、吴用、金翠莲、顾大嫂、戴宗、琼英、李师师、燕青、李俊、Natasha Rostova、Natasha、Rostova、natasha、Marya Bolkonskaya、Marya、Bolkonskaya、玛丽亚、Helene Kuragina、Boris Drubetskoy、Pierre Bezukhov、Murat |
| d6f06ba7 | Annette Scherer、Vera Rostova、Speranski、Langeron、Dokhturov、Ermolov、Ney、Uvarov、Davout、Tormasov、Bourienne、Tushin、Dunyasha、Chichagov、Junot、Pierre Bezukhov、Andrei Bolkonsky、Marya Bolkonskaya、Boris Drubetskoy、Nikolai Rostov、Fedor Dolokhov、Karp Rapp、Pyotr Bagration、Kutuzov、Helene Kuragina、Yuri Zherkov、Vasily Balashev、Balashev、Sergei Wolzogen、Adrian Pfuel、Pfuel、Matvey Weyrother、Weyrother、Stepan Nesvitsky、Nesvitsky、Murat、Julie Karagina、Liza、Platon Karataev、Lev Raevsky、Berthier、Bessieres、Platov、Hippolyte、Kuzma Timokhin、Sonya、Oudinot、Catiche、Mavra、Anatole Dokhturov、Anatole、murat、bessieres、berthier、helene-kuragina、sonyas、liza、oudinot |
| db4da839 | Kutuzov、Liza、Ivan Timokhin、Louis Berthier、Pierre Bezukhov、dolokhov |
| ea89e4d7 | Anatole Kuragin、Anatole、Murat、Platon |
| ea904f1d | 段景住、高俅、穆弘、Kutuzov、Balashev、巴格拉季昂、海伦、多洛霍夫、拉耶夫斯基、罗斯托夫、皮埃尔、Pierre Bezukhov、蔡京 |
| f8caa478 | 武松 |
| eba40c49 | Pyotr Bagration、Adolf Berg、Alexander Bilibin、Fedor Dolokhov、Sonya Rostova、Speranski、Murat、Ney、Massena、Lannes、Duroc、Anna Pavlovna、Soult、Legrand、Merle、Catiche、Thouvenot、Pierre Bezukhov、Andrei Bolkonsky、Marya Bolkonskaya、Vasili Denisov、Boris Drubetskoy、Anatole Kuragin、Hippolyte Kuragin、Platon Karataev、Helene Kuragina、Helene、Kuzma Timokhin、Julie Karagina、Liza、Stepan Nesvitsky、Dunyasha、Vasily Balashev、Annette Scherer、Mavra Kuzminishna、Sergei Wolzogen、Karp Rapp、Pelageya、Fadei Savary、Uvar Friant、Zakhar Berthier、Kutuzov、Agrafena Ivanovna、Bennigsen Jr.、Bennigsen、Lev Raevsky、Tikhon、vasily-balashev、annette-scherer、mavra-kuzminishna、sergei-wolzogen、karp-rapp、fsavary、bennigsen-jr |

---
name: feishu-doc
description: 读写本项目的飞书文档「StarVLA项目」（wiki VWXnwHjN4iVlKUkm7HZcN7Ddnjc）。同步计划、写实验笔记、插图片/GIF/视频到飞书时使用。依赖本机已安装的 lark-cli。
---

# 飞书文档「StarVLA项目」

## 文档信息（2026-08-23 验证可读写）

| 项 | 值 |
|---|---|
| Wiki 链接 | https://my.feishu.cn/wiki/VWXnwHjN4iVlKUkm7HZcN7Ddnjc |
| node_token | `VWXnwHjN4iVlKUkm7HZcN7Ddnjc` |
| 底层 docx token（obj_token） | `XPs1d1chrobe45xZoXCcFI7knRh` |
| space_id | `7497129636623663108` |
| 身份 | `--as user` |

通用语法以全局 skill `lark-doc` 为准，本 skill 只记录本项目特有信息。
姊妹项目「SONIC项目」在同一知识空间（`A26bwAyaviDV3pkelBhcsBzGnfg`），写法一致。

## 常用命令

```bash
# 读整篇
lark-cli docs +fetch --as user --doc "https://my.feishu.cn/wiki/VWXnwHjN4iVlKUkm7HZcN7Ddnjc" --doc-format markdown
# 拿 block id
lark-cli docs +fetch --as user --doc "https://my.feishu.cn/wiki/VWXnwHjN4iVlKUkm7HZcN7Ddnjc" --scope outline --max-depth 2 --detail with-ids
# 文末追加
lark-cli docs +update --as user --doc "https://my.feishu.cn/wiki/VWXnwHjN4iVlKUkm7HZcN7Ddnjc" --command append --content '<h1>标题</h1><p>正文</p>'
# 建子页
lark-cli wiki +node-create --as user --parent-node-token VWXnwHjN4iVlKUkm7HZcN7Ddnjc --title "标题"
# 列子页
lark-cli wiki +node-list --as user --space-id 7497129636623663108 --parent-node-token VWXnwHjN4iVlKUkm7HZcN7Ddnjc
```

## 子页一览

| 页 | node_token | obj_token（docx） |
|---|---|---|
| （建一个补一行） | | |

父页顶部维护「目录」列表，每建一个子页就加一条 `<ul><li><cite type="doc" doc-id="<obj_token>"/> — 一句话说明</li></ul>`，
用 `block_insert_after --block-id <上一个 li 的 id>` 插入。

## 媒体

`docs +media-insert` 只认 docx token（obj_token），不认 wiki URL：
```bash
lark-cli docs +media-insert --as user --doc <obj_token> --file ./x.gif --width 800 --caption "..."
lark-cli docs +media-insert --as user --doc <obj_token> --file ./x.mp4 --type file
```
媒体只能插到文末 → 先写文字再插媒体，或分段 append。GIF 控制在 5 MB 内（抽帧 + 缩放）。

## 规则

- 改已有内容用 `block_replace` / `block_insert_after`，**不要用 `overwrite`**（会丢图片和评论）。
- 每次写操作后 block ID 会变，重新 fetch 再做下一步。
- 写完用 `+fetch --doc-format markdown` 回读验证。
- 网络受限时 lark-cli 会连不上，处理方式同 `git-push` skill。

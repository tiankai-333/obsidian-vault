

---

# Obsidian + Git 版本控制手册（你的库：tiankai-333/obsidian-vault）

> 目标：我只在电脑上用 Git 做版本控制与备份到 GitHub（代码托管平台）。手机不同步。

## 0. 你的当前配置快照（以后照着核对）

- 本地 Vault 路径：`C:\Dev\Docs\Obsidian Vault`
    
- GitHub 用户名：`tiankai-333`
    
- 仓库名：`obsidian-vault`
    
- 远端地址（SSH (Secure Shell，安全外壳协议)）：`git@github.com:tiankai-333/obsidian-vault.git`
    
- 本地分支：`main`
    
- 提交者信息：
    
    - `user.name = Tucker_Wu`
        
    - `user.email = wutiankai179@gmail.com`
        
- SSH 密钥：
    
    - 私钥：`C:\Users\wutia\.ssh\id_ed25519`
        
    - 公钥：`C:\Users\wutia\.ssh\id_ed25519.pub`
        
    - 算法：ED25519（Edwards-curve Digital Signature Algorithm 25519，基于25519椭圆曲线的签名算法）
        

---

## 1. 先把 Git 的“三区域”记牢（理解所有命令的核心）

Git 管文件时，你脑子里只保留这三层：

1. **工作区（Working tree，工作目录）**  
    你在 Obsidian 里真实正在改的文件。
    
2. **暂存区（Staging area / Index，暂存区）**  
    你“决定这次提交要包含哪些改动”的候选清单。
    
3. **提交历史（Commit history，提交历史）**  
    已经落盘的版本快照（可回滚、可对比、可同步到 GitHub）。
    

> 一句话：**add 是把改动放进暂存区；commit 是把暂存区做成历史快照；push/pull 是和 GitHub 同步快照。**

---

## 2. 你每天真正常用的 5 个命令（写 Obsidian 笔记的最小工作流）

### 2.1 `git status` —— 我现在处于什么状态？

用途：随时检查“改了什么、有没有忘记 add/commit”。

你会看到类似信息：

- 当前分支
    
- modified（已修改未暂存）
    
- staged（已暂存待提交）
    
- clean（没有要提交的内容）
    

---

### 2.2 `git add .` —— 把这次所有改动放进“暂存区”

`.` 表示当前目录递归全部改动。  
含义：**把“你现在看到的改动”，加入本次提交候选。**

> 常见提示：  
> `LF will be replaced by CRLF`
> 
> - LF（Line Feed，换行）
>     
> - CRLF（Carriage Return + Line Feed，回车+换行）  
>     这是换行风格提示，一般不影响使用。
>     

---

### 2.3 `git commit -m "update"` —— 把暂存区做成一个版本快照

- commit 会产生一个新的“版本点”（commit id）
    
- `-m` 是这次版本说明（message）
    

**重要规则：commit 只包含“暂存区”的内容。**（没 add 的改动不会进这次 commit）

---

### 2.4 `git push` —— 把本地提交上传到 GitHub

用途：把本地历史同步到远端仓库，做备份/多机共享。

> 你第一次推 main 用过：`git push -u origin main`，`-u` 的作用是设置上游（upstream，上游跟踪），以后直接 `git push` 就行。

---

### 2.5 `git pull` —— 从 GitHub 拉取并合并到本地

用途：如果你以后换电脑、或同仓库在另一台电脑也改过，开始写之前先 pull。  
你之前的“换电脑流程”就是：先 clone，然后日常 `pull -> edit -> commit -> push`。

---

## 3. 推荐你用的“写作型”日常习惯（几乎不会翻车）

### 单机写作（你现在主要就是这个）

写完一段就做一次：

```bat
git status
git add .
git commit -m "update"
git push
```

### 多机写作（以后你可能会用另一台电脑）

开始写之前先：

```bat
git pull
```

写完再走一遍 add/commit/push。

---

## 4. 一次性初始化（你已经做过，但这里写清楚方便复现）

### 4.1 初始化仓库

```bat
git init
```

### 4.2 配置提交者身份（解决 Author identity unknown）

```bat
git config --global user.name "Tucker_Wu"
git config --global user.email "wutiankai179@gmail.com"
```

（这一步是为了解决 commit 时不知道“你是谁”。）

### 4.3 绑定远端并推送 main

```bat
git remote add origin git@github.com:tiankai-333/obsidian-vault.git
git branch -M main
git push -u origin main
```

远端地址与 `-u` 的含义见上文。

---

## 5. SSH（Secure Shell，安全外壳协议）推送为什么会失败、你是怎么修好的

### 5.1 报错：`Permission denied (publickey).`

含义：你用了 SSH 推送，但 GitHub 没有你这台电脑对应的公钥，所以拒绝。  
SSH 的逻辑是“本机私钥证明身份，GitHub 用公钥验证”。

### 5.2 修复流程（你已经做完）

1. 生成密钥对（公钥+私钥）：
    

```bat
ssh-keygen -t ed25519 -C "wutiankai179@gmail.com"
```

会得到 `id_ed25519`（私钥）与 `id_ed25519.pub`（公钥）。

2. 打印公钥并复制到 GitHub 的 SSH keys：
    

```bat
type %USERPROFILE%\.ssh\id_ed25519.pub
```

注意：`type ...pub` 必须在生成完密钥、回到命令行提示符后执行（不要输入到 ssh-keygen 的交互里）。

3. 测试：
    

```bat
ssh -T git@github.com
```

出现 `Hi tiankai-333! ...` 就 OK。

---

## 6. .gitignore（忽略规则）建议（Obsidian 场景）

你至少应该忽略这些“机器状态/缓存”类文件，避免无意义变更：

```gitignore
.obsidian/workspace*
.obsidian/cache
.trash/
.DS_Store
Thumbs.db
```

> 原则：
> 
> - “笔记内容、脚本、模板”可以进 Git
>     
> - “窗口布局、缓存、垃圾桶”一般不进 Git
>     

---

## 7. 新电脑接入同一个仓库（以后用得上）

在新电脑上，核心流程是：**配置 SSH -> clone -> Obsidian 打开文件夹**：

```bat
cd /d C:\Dev\Docs
git clone git@github.com:tiankai-333/obsidian-vault.git
```

然后 Obsidian：Open folder as vault，选择 `C:\Dev\Docs\obsidian-vault`。

---

## 8. 你最可能遇到的 4 个坑（以及处理方式）

1. **忘了 add**：commit 后发现文件没进历史  
    → `git status` 看看哪些文件还没 staged，然后 `git add .` 再 commit。
    
2. **push 被拒绝（远端更新了）**：  
    → 先 `git pull`，如果冲突再处理。
    
3. **LF/CRLF 警告**：  
    → 多数情况可忽略（只是提示）。
    
4. **SSH publickey 报错**：  
    → 按第 5 节重新走一遍（通常是新电脑没加公钥）。
    

---

## 9. 最短记忆版（只记这两行）

- 写之前（可能换电脑/多人）：`git pull`
    
- 写之后：`git add . && git commit -m "update" && git push`
    

# Git 常用命令与 GitHub 项目维护指南

## 1. Git 基本概念

| 名称 | 说明 |
| --- | --- |
| 工作区 | 本地正在编辑的文件目录 |
| 暂存区 | 使用 `git add` 后，等待提交的修改集合 |
| 本地仓库 | 项目中的 `.git` 目录，保存提交历史和分支信息 |
| 远程仓库 | GitHub 上用于备份、同步和协作的仓库 |
| 分支 | 一条独立的开发线，例如 `main`、`feature/login` |
| 提交 | 一个可追踪的代码版本，即 commit |

日常最常见流程：

```text
修改文件 -> git add -> git commit -> git push
获取远程更新 -> git pull -> 继续开发
```

## 2. 首次安装后的身份配置

首次使用 Git 时，配置提交者姓名和邮箱：

```powershell
git config --global user.name "你的 GitHub 用户名"
git config --global user.email "你的邮箱"
git config --global --list
```

## 3. 查看仓库状态与提交历史

```powershell
git status                         # 查看修改、暂存和分支状态
git log --oneline --graph --all    # 简洁查看提交历史与分支关系
git diff                           # 查看尚未暂存的修改
git diff --staged                  # 查看已经暂存的修改
git remote -v                      # 查看远程仓库地址
```

## 4. 提交和推送代码

```powershell
git add 文件名                     # 暂存指定文件
git add .                          # 暂存当前目录下的全部修改
git commit -m "描述本次修改"       # 创建本地提交
git push                           # 将提交推送到远程仓库
```

建议每次提交前先执行：

```powershell
git status
git diff --staged
```

确认没有误提交密钥、大文件、临时结果或数据集。

## 5. 获取远程更新

```powershell
git fetch origin                   # 下载远程信息，但不合并代码
git pull origin main               # 拉取并合并远程 main 分支
git pull                           # 已建立追踪关系时的简写
```

## 6. 分支常用操作

```powershell
git branch                         # 查看本地分支
git branch -a                      # 查看本地和远程分支
git switch -c feature/demo         # 创建并切换到新分支
git switch main                    # 切换到已有分支 main
git push -u origin feature/demo    # 首次推送新分支并建立追踪关系
git merge feature/demo             # 将指定分支合并到当前分支
git branch -d feature/demo         # 删除已合并的本地分支
```

如果远程已有分支，但本地还没有：

```powershell
git fetch origin
git switch --track origin/分支名
```

## 7. 撤销和临时保存修改

### 7.1 撤销操作

```powershell
git restore 文件名                 # 丢弃尚未暂存的文件修改
git restore --staged 文件名        # 取消暂存，但保留文件修改
git commit --amend                 # 修改最近一次提交
git revert 提交ID                  # 通过新提交撤销历史中的某次提交
```

注意：`git restore 文件名` 会丢失该文件尚未提交的修改，执行前应确认内容不再需要。

### 7.2 临时收起未完成工作

```powershell
git stash                          # 临时收起当前修改
git stash list                     # 查看暂存记录
git stash pop                      # 恢复最近一次暂存的修改
```

## 8. 删除或移动受 Git 管理的文件

```powershell
git rm 文件名
git mv 原文件名 新文件名
```

## 9. 将普通本地工程托管到 GitHub

假设本地项目目录是：

```powershell
e:\my_project
```

### 9.1 初始化本地 Git 仓库

```powershell
cd e:\my_project
git init
git branch -M main
```

### 9.2 创建 `.gitignore`

`.gitignore` 用于排除不应提交的文件。Python 项目通常可包含：

```gitignore
__pycache__/
*.pyc
.venv/
venv/
.env
.vscode/
.idea/
dist/
build/
*.log
```

深度学习或计算机视觉项目通常还需要排除数据、权重和输出结果：

```gitignore
checkpoints/
weights/
outputs/
datasets/
*.pth
*.pt
*.ckpt
```

提交前不要上传密码、访问令牌、私钥、大型数据集或不必要的模型文件。

### 9.3 创建首次本地提交

```powershell
git add .
git status
git commit -m "Initial commit"
```

### 9.4 在 GitHub 创建空仓库

登录 GitHub 后创建新仓库。对于已经存在本地文件的项目，建议创建空仓库，不要预先添加 `README`、`.gitignore` 或 License，以避免首次推送时产生历史冲突。

创建后会得到类似以下 HTTPS 地址：

```text
https://github.com/你的用户名/my_project.git
```

### 9.5 关联远程仓库并首次推送

```powershell
git remote add origin https://github.com/你的用户名/my_project.git
git push -u origin main
```

`-u` 会建立本地 `main` 与远程 `origin/main` 的追踪关系。之后可以直接使用：

```powershell
git pull
git push
```

## 10. 日常开发推荐流程

### 10.1 小型个人修改直接提交到 `main`

```powershell
git pull

# 编辑文件

git status
git add .
git commit -m "Add evaluation script"
git push
```

### 10.2 使用新分支开发功能

```powershell
git switch main
git pull
git switch -c feature/new-function

# 编辑文件

git add .
git commit -m "Implement new function"
git push -u origin feature/new-function
```

随后可以在 GitHub 上创建 Pull Request，将新分支合并到 `main`。

## 11. 从 GitHub 克隆已有项目

```powershell
cd e:\
git clone https://github.com/用户名/仓库名.git
cd 仓库名
```

## 12. GitHub 登录与 HTTPS 推送

GitHub 已不接受账号密码作为 Git 命令的认证密码。使用 HTTPS 推送时，通常可以通过 Git Credential Manager 弹出的浏览器窗口登录授权，也可以配置 Personal Access Token。

当前 `vggt` 项目已使用 HTTPS 远程地址。在当前网络环境下，SSH 访问 GitHub 会被中断，因此推荐继续使用 HTTPS：

```powershell
cd e:\vgg_xzy\vggt
git pull
git add .
git commit -m "本次修改说明"
git push
```

## 13. 常用命令速查表

| 目标 | 命令 |
| --- | --- |
| 初始化仓库 | `git init` |
| 查看状态 | `git status` |
| 暂存全部修改 | `git add .` |
| 创建提交 | `git commit -m "说明"` |
| 查看历史 | `git log --oneline --graph --all` |
| 拉取更新 | `git pull` |
| 推送提交 | `git push` |
| 查看分支 | `git branch -a` |
| 新建并切换分支 | `git switch -c 分支名` |
| 切换分支 | `git switch 分支名` |
| 首次推送新分支 | `git push -u origin 分支名` |
| 暂存未完成修改 | `git stash` |
| 恢复暂存修改 | `git stash pop` |
| 查看远程地址 | `git remote -v` |

"""
Hibs Studio - 桌面 GUI
=========================

零配置 Web 界面, 浏览器即开即用.

启动:
    pip install streamlit
    streamlit run hibs_studio/app.py

或者直接:
    python -m hibs_studio
"""
import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import List

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="Hibs Studio",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 自定义 CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 状态管理
# ============================================================
if "data_files" not in st.session_state:
    st.session_state.data_files = []
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "training_status" not in st.session_state:
    st.session_state.training_status = "未开始"


# ============================================================
# 侧边栏导航
# ============================================================
with st.sidebar:
    st.markdown("## 🌪️ Hibs Studio")
    st.caption("个人本地大模型 · 数据不出本机")
    st.divider()

    page = st.radio(
        "导航",
        ["🏠 首页", "📂 数据导入", "🏋️ 训练", "💬 对话", "📱 导出到手机", "⚙️ 设置"],
        index=0,
    )

    st.divider()
    st.caption("v0.1.0 · V16.6 50M")


# ============================================================
# 页面: 首页
# ============================================================
def page_home():
    st.markdown('<p class="main-header">用你自己的数据, 训练你自己的模型</p>', unsafe_allow_html=True)
    st.markdown("**完全本地 · 完全隐私 · 5 分钟从零到对话**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="stat-box"><h3>50M</h3><p>参数</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-box"><h3>200MB</h3><p>模型大小</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-box"><h3>0</h3><p>数据上传</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-box"><h3>∞</h3><p>个人风格</p></div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🚀 快速开始 (3 步)")

    st.markdown("""
    **步骤 1**: 点击左侧 `📂 数据导入`, 上传你的文本文件
    (txt / md / pdf / docx / 微信导出 csv)

    **步骤 2**: 点击 `🏋️ 训练`, 选择风格, 点"开始训练"
    (普通笔记本 30 分钟 - 2 小时)

    **步骤 3**: 点击 `💬 对话`, 用你自己的语言和模型聊天
    """)

    st.info("💡 **提示**: 模型会学习你的写作风格、用词习惯、句式结构. "
            "数据越多、越有代表性, 效果越好.")

    with st.expander("📚 进阶 - 适合什么样的数据?"):
        st.markdown("""
        - ✅ **日记/随笔** (100-1000 篇) - 学习个人语气
        - ✅ **专业笔记** - 学习领域术语
        - ✅ **对话记录** - 学习对话模式
        - ✅ **代码片段** - 学习编程风格
        - ❌ 少于 1 万字的纯短句 - 数据太少
        - ❌ 全是链接和图片 - 没有可学内容
        """)


# ============================================================
# 页面: 数据导入
# ============================================================
def page_data():
    st.markdown("## 📂 数据导入")
    st.caption("把你要学习的所有文本拖进来. 支持 txt / md / pdf / docx / csv")

    # 上传
    uploaded_files = st.file_uploader(
        "拖拽文件到这里, 或点击上传",
        type=["txt", "md", "pdf", "docx", "csv", "json"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # 保存到工作目录
        work_dir = ROOT / "workspace" / "user_data"
        work_dir.mkdir(parents=True, exist_ok=True)

        for f in uploaded_files:
            save_path = work_dir / f.name
            with open(save_path, "wb") as out:
                out.write(f.read())
            st.session_state.data_files.append(str(save_path))

    # 显示已上传
    st.divider()
    st.markdown("### 已导入文件")

    if not st.session_state.data_files:
        st.info("还没有文件, 拖拽文件到上方开始")
    else:
        for f in st.session_state.data_files:
            size = os.path.getsize(f)
            st.markdown(f"- 📄 `{Path(f).name}` ({size/1024:.1f} KB)")

        total_size = sum(os.path.getsize(f) for f in st.session_state.data_files)
        st.success(f"**{len(st.session_state.data_files)} 个文件, 共 {total_size/1024:.1f} KB**")

        # 转换按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 转换为训练格式", use_container_width=True):
                with st.spinner("转换中..."):
                    from hibs_studio.importer import convert_to_training_format
                    output_path = convert_to_training_format(
                        st.session_state.data_files,
                        work_dir / "train.jsonl",
                    )
                    st.success(f"✅ 转换完成: {output_path}")

        with col2:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state.data_files = []
                st.rerun()


# ============================================================
# 页面: 训练
# ============================================================
def page_train():
    st.markdown("## 🏋️ 训练")
    st.caption("用你导入的数据训练个性化模型")

    # 检查数据
    train_file = ROOT / "workspace" / "user_data" / "train.jsonl"
    if not train_file.exists():
        st.warning("⚠️ 请先在 `📂 数据导入` 页面上传文件并转换")
        return

    # 数据统计
    with open(train_file) as f:
        n_lines = sum(1 for _ in f)
    total_chars = train_file.stat().st_size
    st.info(f"📊 训练数据: {n_lines} 段, {total_chars/1024:.1f} KB")

    st.divider()

    # 训练参数
    st.markdown("### 训练设置")

    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider("训练轮数", 1, 20, 5)
        preset = st.selectbox(
            "模型规模",
            ["快速 (5M, 笔记本 10 分钟)", "标准 (50M, 笔记本 1-3 小时)", "深度 (100M, 需 GPU)"],
            index=1,
        )

    with col2:
        learning_rate = st.select_slider(
            "学习率",
            options=[1e-4, 3e-4, 1e-3],
            value=3e-4,
        )
        style = st.selectbox(
            "风格偏好",
            ["保留原风格 (推荐)", "更口语", "更正式", "更简洁"],
        )

    st.divider()

    # 开始按钮
    if st.button("🚀 开始训练", type="primary", use_container_width=True):
        st.session_state.training_status = "训练中..."
        progress = st.progress(0, "初始化...")

        # 简化版: 调用训练脚本 (实际应分批 stream 进度)
        with st.spinner("训练中... 请勿关闭页面"):
            from hibs_studio.trainer import train_personal_model
            model_path = train_personal_model(
                train_file=str(train_file),
                epochs=epochs,
                preset=preset,
                lr=learning_rate,
                progress_callback=lambda p, msg: progress.progress(p, msg),
            )

        st.session_state.training_status = "完成"
        st.success(f"✅ 训练完成! 模型保存到: {model_path}")
        st.balloons()


# ============================================================
# 页面: 对话
# ============================================================
def page_chat():
    st.markdown("## 💬 对话")
    st.caption("用你的个性化模型聊天")

    # 检查模型
    model_path = ROOT / "checkpoints" / "user_model_best.pt"
    if not model_path.exists():
        st.warning("⚠️ 请先训练一个模型 (左侧 `🏋️ 训练`)")
        return

    # 加载模型 (缓存)
    @st.cache_resource
    def load_engine(path):
        from hibs_cli.inference import InferenceEngine
        return InferenceEngine(str(path), device="cpu")

    with st.spinner("加载模型..."):
        engine = load_engine(model_path)

    # 参数
    with st.sidebar:
        st.divider()
        temperature = st.slider("创造性", 0.1, 2.0, 0.8, 0.1)
        max_tokens = st.slider("最大长度", 32, 512, 128, 32)

    # 对话界面
    st.markdown("### 🗨️ 聊天")

    # 显示历史
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # 输入
    if prompt := st.chat_input("输入消息..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = engine.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
            st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# ============================================================
# 页面: 导出到手机
# ============================================================
def page_export_mobile():
    st.markdown("## 📱 导出到手机")
    st.caption("把模型带到你的手机上")

    model_path = ROOT / "checkpoints" / "user_model_best.pt"
    if not model_path.exists():
        st.warning("⚠️ 请先训练模型")
        return

    if st.button("📦 生成手机包", use_container_width=True):
        with st.spinner("导出中..."):
            from hibs_studio.exporter import export_mobile_package
            package_path = export_mobile_package(model_path)

        st.success(f"✅ 导出完成: {package_path}")
        st.info("""
        **使用方式**:
        - iOS: 用 Files 打开 `HibsApp.zip` → 解压后用 AltStore 安装
        - Android: 用 `adb install hibs.apk`
        """)

        with open(package_path, "rb") as f:
            st.download_button(
                "⬇️ 下载到电脑",
                f,
                file_name=Path(package_path).name,
            )


# ============================================================
# 页面: 设置
# ============================================================
def page_settings():
    st.markdown("## ⚙️ 设置")

    st.markdown("### 设备")
    device = st.radio("推理设备", ["自动", "CPU", "GPU (CUDA)"], index=0)
    st.caption(f"当前: {'GPU' if st.session_state.get('use_cuda', False) else 'CPU'}")

    st.divider()

    st.markdown("### 高级")
    quantization = st.checkbox("INT8 量化 (减小模型大小)", value=True)
    if quantization:
        st.caption("量化后模型 ~50MB, 精度可能略有下降")

    st.divider()

    st.markdown("### 数据管理")
    if st.button("🗑️ 清空所有数据", type="secondary"):
        work_dir = ROOT / "workspace"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        st.success("已清空")


# ============================================================
# 路由
# ============================================================
PAGES = {
    "🏠 首页": page_home,
    "📂 数据导入": page_data,
    "🏋️ 训练": page_train,
    "💬 对话": page_chat,
    "📱 导出到手机": page_export_mobile,
    "⚙️ 设置": page_settings,
}

PAGES[page]()


# ============================================================
# CLI 入口
# ============================================================
def main():
    """CLI 入口: python -m hibs_studio"""
    os.system(f"streamlit run {__file__} --server.port 8501 --server.address 0.0.0.0")


if __name__ == "__main__":
    main()
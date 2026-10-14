"""TurboQuant Indic Document Assistant - Streamlit Dashboard.

Provides long-context Indic document Question-Answering with zero-copy prefix rewinding
and live KV-cache memory telemetry comparing TurboQuant 2-bit vs. uncompressed bfloat16.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

from turboquant.cache import TurboQuantCache
from turboquant.ingest import DocumentPrefixEngine, PrefixState
from turboquant.qa import IndicDocumentQAEngine, QAResult
from turboquant.quantizer import DynamicKVQuantizer

# ==============================================================================
# Sample Indic Documents
# ==============================================================================

SAMPLE_DOCS: Dict[str, str] = {
    "Hindi - चंद्रयान-3 और अंतरिक्ष अनुसंधान": (
        "चंद्रयान-3 भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) का एक प्रमुख अंतरिक्ष मिशन है। "
        "यह 14 जुलाई 2023 को सतीश धवन अंतरिक्ष केंद्र, श्रीहरिकोटा से प्रक्षेपित किया गया था। "
        "23 अगस्त 2023 को चंद्रयान-3 के विक्रम लैंडर ने चंद्रमा के दक्षिणी ध्रुव पर सफल सॉफ्ट लैंडिंग की। "
        "इसके साथ ही भारत चंद्रमा के दक्षिणी ध्रुव पर पहुँचने वाला दुनिया का पहला देश बन गया। "
        "मिशन के मुख्य घटकों में एक प्रणोदन मॉड्यूल, विक्रम लैंडर और प्रज्ञान रोवर शामिल थे। "
        "प्रज्ञान रोवर ने चंद्रमा की सतह पर खनिजों और तत्वों की उपस्थिति की पुष्टि की, विशेष रूप से सल्फर की खोज की। "
        "इसरो का मुख्यालय बेंगलुरु, कर्नाटक में स्थित है और इसके वर्तमान अध्यक्ष डॉ. एस. सोमनाथ हैं। "
        "चंद्रयान-3 की ऐतिहासिक सफलता के दिन को भारत सरकार ने 'राष्ट्रीय अंतरिक्ष दिवस' के रूप में घोषित किया।"
    ),
    "Tamil - தமிழ் செம்மொழி வரலாறு மற்றும் சங்க இலக்கியம்": (
        "தமிழ் மொழி உலகின் மிகத் தொன்மையான செம்மொழிகளில் ஒன்றாகும். "
        "இது திராவிட மொழிக் குடும்பத்தைச் சேர்ந்தது மற்றும் இரண்டாயிரத்திற்கும் மேற்பட்ட ஆண்டுகள் பழமையான இலக்கிய வரலாற்றைக் கொண்டுள்ளது. "
        "சங்க இலக்கியம் என்பது கி.மு. 300 முதல் கி.பி. 300 வரையிலான காலப்பகுதியில் எழுதப்பட்ட எட்டுத்தொகை மற்றும் பத்துப்பாட்டு நூல்களைக் குறிக்கிறது. "
        "திருவள்ளுவர் எழுதிய திருக்குறள் உலகப் பொதுமறை என்று அழைக்கப்படுகிறது, இதில் 1330 குறட்பாக்கள் மூன்று பால்களாகப் பிரிக்கப்பட்டுள்ளன: "
        "அறத்துப்பால், பொருட்பால், மற்றும் காமத்துப்பால். "
        "2004 ஆம் ஆண்டில் இந்திய அரசால் முதன்முதலில் செம்மொழி தகுதி வழங்கப்பட்ட மொழி தமிழ் ஆகும். "
        "தமிழர்களின் கப்பற்கலை, சோழர்களின் கடல்வழி வணிகம் மற்றும் தஞ்சைப் பெரிய கோவில் ஆகியவை தமிழர்களின் கட்டிடக்கலை மற்றும் வரலாற்றுச் சிறப்பைப் பறைசாற்றுகின்றன."
    ),
    "Telugu - విజయనగర సామ్రాజ్యం మరియు సాహిత్యం": (
        "విజయనగర సామ్రాజ్యం దక్షిణ భారతదేశ చరిత్రలో అత్యంత ప్రసిద్ధి చెందిన మహోన్నత సామ్రాజ్యాలలో ఒకటి. "
        "దీనిని 1336 సంవత్సరంలో హరిహర రాయలు మరియు బుక్క రాయలు తుంగభద్ర నదీ తీరంలో స్థాపించారు. "
        "శ్రీకృష్ణదేవరాయల పాలనా కాలం (1509-1529) విజయనగర చరిత్రలో స్వర్ణయుగంగా పరిగణించబడుతుంది. "
        "ఆయన స్వయంగా గొప్ప కవి మరియు 'ఆముక్తమాల్యద' అనే ప్రసిద్ధ తెలుగు ప్రబంధాన్ని రచించారు. "
        "ఆయన ఆస్థానంలో 'అష్టదిగ్గజాలు' అని పిలువబడే ఎనిమిది మంది ప్రముఖ కవులు ఉండేవారు, వారిలో అల్లసాని పెద్దన, తెనాలి రామకృష్ణ ముఖ్యులు. "
        "హంపి విజయనగర సామ్రాజ్యానికి రాజధానిగా విలసిల్లింది, ఇది ప్రస్తుతం యునెస్కో ప్రపంచ వారసత్వ ప్రదేశంగా గుర్తించబడింది."
    ),
}


# ==============================================================================
# Pure Helper Logic (Testable & Decoupled from Streamlit UI)
# ==============================================================================


def compute_doc_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of document text to track changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_telemetry_metrics(
    prefix_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    cache_bytes: int,
    prefill_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """Calculate uncompressed bfloat16 baseline vs TurboQuant KV cache memory and speed.

    Uncompressed bfloat16 baseline formula:
        2 (K & V) * num_layers * num_kv_heads * head_dim * prefix_length * 2 bytes (bfloat16)

    Args:
        prefix_length: Total number of ingested document tokens (L_doc).
        num_layers: Number of transformer hidden layers.
        num_kv_heads: Number of key-value attention heads (GQA).
        head_dim: Dimensionality of each attention head vector (e.g. 128 or 64).
        cache_bytes: Actual memory consumed by TurboQuantCache in bytes.
        prefill_time_sec: Wall-clock seconds elapsed during document ingestion.

    Returns:
        Dictionary containing baseline_bytes, turboquant_bytes, memory_saved_bytes,
        memory_saved_mib, memory_saved_pct, compression_ratio, prefill_tokens_per_sec,
        turboquant_mib, and baseline_mib.
    """
    baseline_bytes = 2 * num_layers * num_kv_heads * head_dim * prefix_length * 2
    turboquant_bytes = cache_bytes
    memory_saved_bytes = max(0, baseline_bytes - turboquant_bytes)
    memory_saved_mib = memory_saved_bytes / (1024 * 1024)
    turboquant_mib = turboquant_bytes / (1024 * 1024)
    baseline_mib = baseline_bytes / (1024 * 1024)

    if baseline_bytes > 0:
        memory_saved_pct = (memory_saved_bytes / baseline_bytes) * 100.0
        compression_ratio = baseline_bytes / turboquant_bytes if turboquant_bytes > 0 else 0.0
    else:
        memory_saved_pct = 0.0
        compression_ratio = 1.0

    prefill_tps = (prefix_length / prefill_time_sec) if prefill_time_sec > 0 else 0.0

    return {
        "baseline_bytes": baseline_bytes,
        "turboquant_bytes": turboquant_bytes,
        "memory_saved_bytes": memory_saved_bytes,
        "memory_saved_mib": memory_saved_mib,
        "memory_saved_pct": memory_saved_pct,
        "compression_ratio": compression_ratio,
        "prefill_tokens_per_sec": prefill_tps,
        "turboquant_mib": turboquant_mib,
        "baseline_mib": baseline_mib,
        "prefix_length": prefix_length,
    }


class SimpleSyntheticTokenizer:
    """Fast, deterministic character-mapping tokenizer for synthetic demo mode."""

    def __init__(self, vocab_size: int = 512, eos_token_id: int = 2) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: bool = False,
    ) -> Dict[str, Any]:
        ids = [3 + (ord(c) % (self.vocab_size - 4)) for c in text]
        if not ids:
            ids = [3]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: bool = False,
    ) -> Union[torch.Tensor, List[int]]:
        res = self(text, return_tensors=return_tensors, add_special_tokens=add_special_tokens)
        return res["input_ids"]

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []
        for t in token_ids:
            if skip_special_tokens and t == self.eos_token_id:
                continue
            chars.append(chr(65 + (t % 26)))
        return "".join(chars)


def create_synthetic_model(
    vocab_size: int = 512,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 32,
) -> Tuple[nn.Module, Any]:
    """Instantiate a lightweight synthetic LLaMA causal model for demo and unit testing."""
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        pad_token_id=0,
        eos_token_id=2,
        attention_bias=False,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    tokenizer = SimpleSyntheticTokenizer(vocab_size=vocab_size, eos_token_id=2)
    return model, tokenizer


def ingest_document_if_needed(
    document_text: str,
    model: nn.Module,
    tokenizer: Any,
    quantizer: DynamicKVQuantizer,
    chunk_size: int = 512,
    current_hash: Optional[str] = None,
    current_state: Optional[PrefixState] = None,
) -> Tuple[PrefixState, str, bool]:
    """Ingest document into TurboQuantCache if not already ingested or text has changed.

    Returns:
        Tuple of (PrefixState, doc_hash, did_ingest_boolean)
    """
    new_hash = compute_doc_hash(document_text)
    if current_state is not None and current_hash == new_hash:
        return current_state, current_hash, False

    engine = DocumentPrefixEngine(
        model=model,
        tokenizer=tokenizer,
        quantizer=quantizer,
        chunk_size=chunk_size,
    )
    new_state = engine.ingest(document_text)
    return new_state, new_hash, True


def process_question(
    qa_engine: IndicDocumentQAEngine,
    prefix_state: PrefixState,
    question: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> QAResult:
    """Answer question against prefix_state with automatic zero-copy prefix rewinding."""
    return qa_engine.answer(
        prefix_state=prefix_state,
        question=question,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# ==============================================================================
# Streamlit Application Setup & Cached Resources
# ==============================================================================

if st is not None:

    @st.cache_resource
    def load_cached_model(
        demo_mode: bool,
        model_id: str = "sarvamai/sarvam-1",
    ) -> Tuple[nn.Module, Any, int, int, int]:
        """Load and cache model in CPU/GPU memory."""
        if demo_mode:
            model, tokenizer = create_synthetic_model()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            model.eval()

        config = getattr(model, "config", None)
        n_layers = getattr(config, "num_hidden_layers", 2)
        h_q = getattr(config, "num_attention_heads", 4)
        h_kv = getattr(config, "num_key_value_heads", h_q)
        head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 128) // h_q)

        return model, tokenizer, n_layers, h_kv, head_dim


def run_app() -> None:
    """Main Streamlit execution loop."""
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install via 'uv add streamlit'.")

    st.set_page_config(
        page_title="TurboQuant Indic Document Assistant",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Header Styling
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e293b;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚡ TurboQuant Indic Document Assistant")
    st.caption("Zero-Copy Prefix Rewinding (Crop) & 2-bit Compressed KV Cache for Long-Context Indic Documents")

    # ==========================================================================
    # Sidebar Controls & Telemetry
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration & Hardware")

        demo_mode = st.toggle(
            "Demo / Synthetic Mode",
            value=True,
            help="When enabled, runs a lightweight synthetic model on CPU without downloading external weights.",
        )

        model_id = "sarvamai/sarvam-1"
        if not demo_mode:
            model_id = st.text_input("HuggingFace Model ID", value="sarvamai/sarvam-1")

        # Load cached model
        with st.spinner("Initializing model..."):
            model, tokenizer, num_layers, num_kv_heads, head_dim = load_cached_model(
                demo_mode=demo_mode, model_id=model_id
            )

        st.subheader("Compression & Decoding")
        use_qjl = st.toggle("1-bit QJL Residual Correction", value=False)
        chunk_size = st.slider("Chunk Size (Tokens)", min_value=64, max_value=1024, value=512, step=64)
        max_new_tokens = st.slider("Max New Tokens", min_value=16, max_value=256, value=64, step=16)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        quantizer = DynamicKVQuantizer(head_dim=head_dim, use_qjl=use_qjl)

        st.divider()
        st.subheader("📊 Live KV-Cache Telemetry")

        # Telemetry Display Area
        telemetry_placeholder = st.empty()

    # ==========================================================================
    # Session State Initialization
    # ==========================================================================
    if "prefix_state" not in st.session_state:
        st.session_state.prefix_state = None
    if "doc_hash" not in st.session_state:
        st.session_state.doc_hash = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "telemetry" not in st.session_state:
        st.session_state.telemetry = None

    # ==========================================================================
    # Document Input Section
    # ==========================================================================
    with st.expander("📄 Document Input & Context Loader", expanded=(st.session_state.prefix_state is None)):
        sample_choice = st.selectbox("Load Sample Indic Document", options=["None"] + list(SAMPLE_DOCS.keys()))

        default_text = ""
        if sample_choice in SAMPLE_DOCS:
            default_text = SAMPLE_DOCS[sample_choice]

        doc_text = st.text_area(
            "Enter Document Text (Hindi, Tamil, Telugu, etc.):",
            value=default_text,
            height=160,
            placeholder="यहाँ अपना दस्तावेज़ पेस्ट करें...",
        )

        col_ingest, col_clear = st.columns([1, 4])
        with col_ingest:
            ingest_btn = st.button("📥 Ingest Document", type="primary")

    # Ingest document if requested or if text is provided and state is empty
    should_ingest = ingest_btn or (doc_text.strip() and st.session_state.prefix_state is None)
    if should_ingest and doc_text.strip():
        with st.spinner("Ingesting document prefix in compressed chunks..."):
            new_state, new_hash, did_ingest = ingest_document_if_needed(
                document_text=doc_text,
                model=model,
                tokenizer=tokenizer,
                quantizer=quantizer,
                chunk_size=chunk_size,
                current_hash=st.session_state.doc_hash,
                current_state=st.session_state.prefix_state,
            )
            st.session_state.prefix_state = new_state
            st.session_state.doc_hash = new_hash

            # Calculate and store telemetry
            telemetry = compute_telemetry_metrics(
                prefix_length=new_state.prefix_length,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                cache_bytes=new_state.cache_bytes,
                prefill_time_sec=new_state.prefill_time_sec,
            )
            st.session_state.telemetry = telemetry

            if did_ingest:
                st.success(
                    f"Document prefix ingested! Length: {new_state.prefix_length} tokens | "
                    f"Prefill speed: {telemetry['prefill_tokens_per_sec']:.1f} tokens/s"
                )

    # Render Sidebar Telemetry
    if st.session_state.telemetry is not None:
        telem = st.session_state.telemetry
        with telemetry_placeholder.container():
            st.metric("Document Tokens (L_doc)", f"{telem['prefix_length']:,}")
            st.metric(
                "TurboQuant KV Footprint",
                f"{telem['turboquant_mib']:.2f} MiB",
                delta=f"-{telem['memory_saved_pct']:.1f}% vs bfloat16",
            )
            st.metric("bfloat16 Baseline Footprint", f"{telem['baseline_mib']:.2f} MiB")
            st.metric("Memory Saved", f"{telem['memory_saved_mib']:.2f} MiB")
            st.metric("Prefill Throughput", f"{telem['prefill_tokens_per_sec']:.1f} tok/s")
    else:
        with telemetry_placeholder.container():
            st.info("Ingest a document to view live KV cache memory telemetry.")

    # ==========================================================================
    # Interactive Chat Loop
    # ==========================================================================
    st.subheader("💬 Document Q&A Chat")

    # Render existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg:
                st.caption(msg["metadata"])

    # Accept user question via chat input
    user_prompt = st.chat_input("दस्तावेज़ से प्रश्न पूछें (Ask a question about the document)...")
    if user_prompt:
        if st.session_state.prefix_state is None:
            st.warning("कृपया पहले दस्तावेज़ इनपुट करें और इनजेस्ट करें। (Please ingest a document first).")
        else:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Generate Answer using QA Engine
            qa_engine = IndicDocumentQAEngine(
                model=model,
                tokenizer=tokenizer,
                quantizer=quantizer,
            )

            with st.chat_message("assistant"):
                with st.spinner("Generating answer with zero-copy prefix rewinding..."):
                    res = process_question(
                        qa_engine=qa_engine,
                        prefix_state=st.session_state.prefix_state,
                        question=user_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )

                meta_text = (
                    f"⏱️ Latency: {res.latency_sec * 1000.0:.1f} ms | "
                    f"⚡ Speed: {res.tokens_per_sec:.1f} tok/s | "
                    f"📝 Tokens: {res.tokens_generated} | "
                    f"💾 KV Cache: {res.cache_bytes / (1024 * 1024):.2f} MiB"
                )

                st.markdown(res.answer)
                st.caption(meta_text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": res.answer,
                        "metadata": meta_text,
                    }
                )


if __name__ == "__main__":
    run_app()

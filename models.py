import io
import logging
from PIL import Image
from transformers import (
    pipeline,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from langchain_huggingface import HuggingFacePipeline

logger = logging.getLogger(__name__)
MODEL_PATH = "/models/medgemma"

# variabili globali
llm = None
image_model = None
processor = None

def init_models():
    global llm, image_model, processor

    # — TEXT MODEL — 
    logger.info("▶️  [TEXT] Loading text-generation pipeline")
    text_pipe = pipeline(
        "text-generation",
        model=MODEL_PATH,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    llm = HuggingFacePipeline(pipeline=text_pipe)
    logger.info("✅  [TEXT] Pipeline ready")

    # — IMAGE-TO-TEXT COMPONENTS — 
    logger.info("▶️  [IMAGE] Loading image-to-text components")
    image_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    logger.info("✅  [IMAGE] Model loaded")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    logger.info("✅  [IMAGE] Processor ready")


def predict_text(prompt: str, max_new_tokens: int = 100) -> str:
    if llm is None:
        raise RuntimeError("Text model not initialized")
    logger.debug(f"predict_text: prompt={prompt!r}, max_new_tokens={max_new_tokens}")
    out = llm.run(prompt, max_new_tokens=max_new_tokens)
    logger.debug(f"predict_text: result={out!r}")
    return out


def predict_image(image_bytes: bytes, user_prompt: str) -> str:
    """
    Inferenza multimodale: immagine + prompt → output testuale (filtrato).
    Il prompt NON deve contenere <start_of_image>, lo aggiungiamo qui.
    """
    if image_model is None or processor is None:
        raise RuntimeError("Image model not initialized")

    logger.info("🖼️  predict_image: start")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.debug(f"predict_image: opened image size={img.size}")
    except Exception:
        logger.exception("predict_image: failed to open image")
        raise

    # Prompt completo da inviare al modello
    full_prompt = (
        "<start_of_image>\n"
        f"{user_prompt.strip()}\n"
        "Findings:\n"
    )
    logger.debug(f"predict_image: full_prompt={full_prompt!r}")

    try:
        inputs = processor(
            text=full_prompt,
            images=img,
            return_tensors="pt",
        ).to(image_model.device)
        logger.debug("predict_image: processor output ready")
    except Exception:
        logger.exception("predict_image: error in processor(...)")
        raise

    try:
        out = image_model.generate(
            **inputs,
            max_new_tokens=300,
            num_beams=3,
            do_sample=False,
            repetition_penalty=1.2,
        )
        logger.debug("predict_image: model.generate done")
    except Exception:
        logger.exception("predict_image: generation error")
        raise

    try:
        decoded = processor.batch_decode(out, skip_special_tokens=True)
        raw = decoded[0].strip()

        # FILTRO: taglia l’eco del prompt e restituisce solo il risultato utile
        for marker in ["Primary Findings:", "Findings:", "Alzheimer's Indicators:"]:
            if marker in raw:
                result = raw[raw.find(marker):].strip()
                break
        else:
            result = raw  # se non trova marker, restituisce tutto

        logger.info(f"🖼️  predict_image: done, result={result!r}")
        return result
    except Exception:
        logger.exception("predict_image: error during decode")
        raise

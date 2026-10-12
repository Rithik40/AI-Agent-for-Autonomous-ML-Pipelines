"""
Main entry point — initialises bot with ConversationHandler for model selection and inference mode.
"""
import socket

# Monkey-patch getaddrinfo to force IPv4. This fixes 'httpx.ConnectError' 
# caused by broken IPv6 routing on Windows with asyncio.
_original_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    results = _original_getaddrinfo(*args, **kwargs)
    return [res for res in results if res[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_getaddrinfo

import logging
import sys

from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ConversationHandler, MessageHandler, filters,
)

import config
from bot.handlers import (
    cmd_start, cmd_help, cmd_exit,
    handle_document, handle_model_selection, handle_tuning_decision, handle_inference_chat, handle_error,
    SELECTING_MODELS, INFERENCE_MODE, WIZARD_PREDICTION, WANT_TUNING,
    handle_wizard_prediction,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    config.validate_config()
    logger.info("Starting ML Pipeline Bot…")

    app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .read_timeout(60)
        .write_timeout(60)
        .connect_timeout(60)
        .build()
    )

    # Commands (outside conversation)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))

    # ConversationHandler:
    #   State 0 (entry): user uploads CSV → AI analysis → show model keyboard
    #   State SELECTING_MODELS: user toggles models → Confirm → run pipeline
    #   State INFERENCE_MODE: pipeline done → bot awaits plain-text prediction queries
    conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Document.ALL, handle_document),
        ],
        states={
            SELECTING_MODELS: [
                CallbackQueryHandler(handle_model_selection),
            ],
            INFERENCE_MODE: [
                # /exit exits inference mode
                CommandHandler("exit", cmd_exit),
                # /predict starts wizard mode
                CommandHandler("predict", handle_inference_chat),
                # any plain text → run live prediction
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_inference_chat),
                # if user uploads a new CSV during inference, restart pipeline
                MessageHandler(filters.Document.ALL, handle_document),
            ],
            WIZARD_PREDICTION: [
                CommandHandler("exit", cmd_exit),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_wizard_prediction),
            ],
            WANT_TUNING: [
                CallbackQueryHandler(handle_tuning_decision),
            ],
        },
        fallbacks=[
            CommandHandler("start", cmd_start),
            CommandHandler("exit",  cmd_exit),
        ],
        per_chat=True,
        per_message=False,
        per_user=True,
    )
    app.add_handler(conv_handler)

    # Global error handler
    app.add_error_handler(handle_error)

    logger.info("Bot is polling. Press Ctrl+C to stop.")
    
    import time
    from telegram.error import NetworkError
    
    retry_delay = 5.0
    while True:
        try:
            app.run_polling(allowed_updates=["message", "callback_query"], close_loop=False)
            break
        except NetworkError as e:
            logger.warning(
                f"Network connection failed during initialization: {e}. "
                f"Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, 30.0)


if __name__ == "__main__":
    main()

import logging

from mani_skill.utils.logging_utils import CustomFormatter, _configure_logger, logger


def remove_logger(name):
    candidate = logging.getLogger(name)
    candidate.handlers.clear()
    logging.Logger.manager.loggerDict.pop(name, None)


def test_package_logger_uses_package_namespace():
    assert logger.name == "mani_skill"


def test_configure_logger_adds_local_handler_when_root_has_handler(capsys):
    name = "mani_skill.tests.root_handler"
    root_handler = logging.NullHandler()
    logging.getLogger().addHandler(root_handler)
    remove_logger(name)

    try:
        configured = _configure_logger(name)

        assert configured.propagate is False
        assert len(configured.handlers) == 1
        assert isinstance(configured.handlers[0], logging.StreamHandler)
        assert isinstance(configured.handlers[0].formatter, CustomFormatter)
        configured.info("local handler is active")
        assert "local handler is active" in capsys.readouterr().err
    finally:
        remove_logger(name)
        logging.getLogger().removeHandler(root_handler)


def test_configure_logger_does_not_duplicate_handlers():
    name = "mani_skill.tests.idempotent"
    remove_logger(name)

    try:
        first = _configure_logger(name)
        initial_handler = first.handlers[0]
        second = _configure_logger(name)

        assert first is second
        assert second.handlers == [initial_handler]
    finally:
        remove_logger(name)

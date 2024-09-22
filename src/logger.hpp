#ifndef LOGGER_H
#define LOGGER_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h> // Подключение для логирования в файл
#include <spdlog/sinks/stdout_color_sinks.h> // Подключение для логирования в консоль
#include <string>

// Объект логгера
inline spdlog::logger logger(
    "multi_sink",
    {std::make_shared<spdlog::sinks::stdout_color_sink_mt>(),
     std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/app.log", false)}
);

// Функция для инициализации логгера
/**
 * @brief Инициализирует логгер
 *
 * @details
 * - Устанавливает шаблон вывода
 * - Выводит строку из 500 символов '-'
 * - Устанавливает формат вывода логов:
 *   [%d-%m-%Y %H:%M:%S.%e] [%^%l%$] %v
 *   - дата, месяц, год, час, минуты, секунды, миллисекунды
 *   - заглавные уровни
 *   - текст сообщения
 */
void init_logger() {
    logger.set_pattern("%v");
    auto print_dashes = [](int count) {
        return std::string(count, '-');
    };
    int num_dashes = 500;
    logger.info(print_dashes(num_dashes));

    // Настройка формата вывода логов
    logger.set_pattern("[%d-%m-%Y %H:%M:%S.%e] [%^%l%$] %v");  // Заглавные уровни, время с миллисекундами
}

#endif LOGGER_H
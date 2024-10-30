#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>

std::string getCurrentDateTimeISO8601ExtendedLocalTime() {
    // Get current time point in system clock
    auto now = std::chrono::system_clock::now();

    // Convert to time_t for local time manipulation
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    // Get local time structure
    std::tm tm_local;
    localtime_s(&tm_local ,&now_c);

    // Get timezone offset in seconds
    std::tm tm_gmt;
    gmtime_s(&tm_gmt, &now_c); // Use gmtime_s instead of gmtime
    int timezone_offset = (std::mktime(&tm_local) - std::mktime(&tm_gmt));

    // Format date and time into ISO 8601 extended format with timezone offset
    std::ostringstream oss;
    oss << std::put_time(&tm_local, "%Y-%m-%dT%H:%M:%S");
    oss << (timezone_offset >= 0 ? "+" : "-");
    oss << std::setw(2) << std::setfill('0') << std::abs(timezone_offset / 3600);
    oss << ":" << std::setw(2) << std::setfill('0') << std::abs((timezone_offset % 3600) / 60);

    return oss.str();
}

int main() {
    std::string currentDateTime = getCurrentDateTimeISO8601ExtendedLocalTime();
    std::cout << "Current date and time (ISO 8601 extended, local timezone): "
        << currentDateTime << std::endl;
    return 0;
}

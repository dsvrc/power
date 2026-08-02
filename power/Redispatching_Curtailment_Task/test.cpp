#include <ctime>

int main() {
    timespec ts;
    timespec_get(&ts, TIME_UTC);
}

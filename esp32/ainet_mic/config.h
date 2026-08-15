#pragma once

#if __has_include("config.local.h")
#include "config.local.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "SET_ME"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS "SET_ME"
#endif

// LAN IPv4 of the Windows PC running AINet (printed when the server starts).
#ifndef AINET_HOST
#define AINET_HOST "SET_ME"
#endif
#ifndef AINET_PORT
#define AINET_PORT 1111
#endif

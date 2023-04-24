#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#define HTTP_VERSION "HTTP/1.1"
#define HTTP_BUFFER_SIZE 4096
#define HTTP_CONNETION_TYPE "Connection: close\r\n"

static std::atomic<int> g_authority(-2);
static std::mutex http_request_lock;

inline char* host_to_ip(const char* hostname) {
    struct hostent* host_entry = gethostbyname(hostname);

    if (host_entry) {
        return inet_ntoa(*(struct in_addr*)*host_entry->h_addr_list);
    }

    return NULL;
}

inline const char* http_send_request(
        const char* hostname,
        uint16_t port,
        const char* resource) {
    char* ip = host_to_ip(hostname);
    if (ip == NULL) {
        return NULL;
    }

    struct sockaddr_in sin = {0};
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);
    sin.sin_addr.s_addr = inet_addr(ip);

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    fcntl(sockfd, F_SETFL, O_NONBLOCK);

    connect(sockfd, (struct sockaddr*)&sin, sizeof(struct sockaddr_in));

    fd_set fdread;
    FD_ZERO(&fdread);
    FD_SET(sockfd, &fdread);
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;

    int so_error = 1;
    if (select(sockfd + 1, NULL, &fdread, NULL, &tv) == 1) {
        socklen_t len = sizeof(so_error);
        getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &so_error, &len);
    }
    if (so_error != 0) {
        close(sockfd);
        return NULL;
    }

    char buffer[HTTP_BUFFER_SIZE] = {0};
    snprintf(buffer,
            HTTP_BUFFER_SIZE,
            "GET %s %s\r\nHost: %s\r\n%s\r\n\r\n",
            resource,
            HTTP_VERSION,
            hostname,
            HTTP_CONNETION_TYPE);

    send(sockfd, buffer, strlen(buffer), 0);

    char* result = reinterpret_cast<char*>(malloc(sizeof(int)));
    memset(result, 0, sizeof(int));
    while (1) {
        int selection = select(sockfd + 1, &fdread, NULL, NULL, &tv);
        if (!selection || !FD_ISSET(sockfd, &fdread)) {
            return NULL;
        } else {
            memset(buffer, 0, HTTP_BUFFER_SIZE);
            int len = recv(sockfd, buffer, HTTP_BUFFER_SIZE, 0);
            if (len == 0) {
                break;
            }
            void* tmp = realloc(result, (strlen(result) + len + 1) * sizeof(char));
            if (NULL == tmp) {
                return NULL;
            } else {
                result = reinterpret_cast<char*>(tmp);
            }
            strncat(result, buffer, len);
        }
    }

    return result;
}

inline std::pair<std::string, uint16_t> get_ip_port(std::string host_name) {
    auto colon_pos = host_name.find(":");
    uint16_t port = 80;
    auto ip = host_name.substr(0, colon_pos);

    if (colon_pos != host_name.npos) {
        port = stoi(host_name.substr(colon_pos + 1));
    }

    return std::pair<std::string, uint16_t>(ip, port);
}

inline void authority_help(std::string ip, uint16_t port, std::string env_resource) {
    /* {"success":true,"message":"ok","code":"200"} */
    const char* response = http_send_request(ip.c_str(), port, env_resource.c_str());

    if (response == NULL) {
        return;
    }

    auto re = std::string(response);
    auto re_http_code = re.substr(9, 1).c_str();
    auto re_body_code = re.substr(re.find("code") + 7, 1).c_str();

    if (strcmp(re_http_code, "2") == 0 && strcmp(re_body_code, "2") == 0) {
        g_authority.store(0);
    }
}

inline void check_authority() {
    std::lock_guard<std::mutex> lock(http_request_lock);

    if (g_authority.load() == -2) {
        std::string uri = "/api/cce/ai-service/v1/auth/token?token=";
        std::map<std::string, std::string> host2token;
        host2token.insert(std::pair<std::string, std::string>(
                    "10.126.82.192:8695",
                    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                    "eyJhdWQiOiJjY2UiLCJleHAiOjQ4MjU3Mjk4NDIsImlhdCI6MTY3MjEyOTg0MiwiaX"
                    "NzIjoiY2xvdWQtbmF0aXZlLWFpIiwic3ViIjoiYWlhayJ9."
                    "bm1T6CW7CmSNsSItk8jp5ODbxDBF7Ia8PrOTuv-c2Yo"));
        host2token.insert(std::pair<std::string, std::string>(
                    "cce-ai-auth-proxy.kube-system.svc",
                    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                    "eyJhdWQiOiJjY2UiLCJleHAiOjQ4MjU3Mjk4NDIsImlhdCI6MTY3MjEyOTg0MiwiaX"
                    "NzIjoiY2xvdWQtbmF0aXZlLWFpIiwic3ViIjoiYWlhayJ9."
                    "bm1T6CW7CmSNsSItk8jp5ODbxDBF7Ia8PrOTuv-c2Yo"));

        for (auto key : host2token) {
            std::string env_resource = uri + key.second;
            auto pair = get_ip_port(key.first);
            std::string ip = pair.first;
            uint16_t port = pair.second;
            authority_help(ip, port, env_resource);
            if (g_authority.load() == 0) {
                break;
            }
        }
    }

    if (g_authority.load() == 0) {
        return;
    } else if (g_authority.load() == -1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }

    g_authority.store(-1);
}

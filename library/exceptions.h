
#pragma once

#include <exception>
#include <string>
#include "holders.h"
#include "utils.h"

class BadBroadCastException : public std::exception {
private:
    std::string message;
public:

    BadBroadCastException(sm_size *shapeOne, int ndimOne, sm_size *shapeTwo, int ndimTwo) {
        char *textOne = convert_shape_to_string(shapeOne, ndimOne);
        char *textTwo = convert_shape_to_string(shapeTwo, ndimTwo);
        message = "Cannot broadcast from shape ";
        message += textOne;
        message += " to shape ";
        message += textTwo;

        free(textOne);
        free(textTwo);
    }

    const char *what() const override {
        return message.c_str();
    }
};


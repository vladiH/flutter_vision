package com.vladih.computer_vision.flutter_vision.utils;

public class responses {
    String type, message;
    public String getMessage() {
        return message;
    }
    public String getType (){
        return  type;
    }
    public static responses success (String message){
        return new Success(message);
    }
    public static responses error (String message){
        return new Error(message);
    }
    public static responses warning (String message){
        return new Warning(message);
    }
    public static responses unexpected (String message){
        return new Unexpected(message);
    }
}

enum ErrorType { success, warning, error, unexpected }

class Success extends responses {
    Success( String message) {
        type = ErrorType.success.name().toString();
        message = message;
    }
}

class Warning extends responses {
    Warning(String message) {
        type = ErrorType.warning.name().toString();
        message = message;
    }
}

class Error extends responses {
    Error(String message) {
        type = ErrorType.error.name().toString();
        message = message;
    }
}

class Unexpected extends responses {
    Unexpected(String message) {
        type = ErrorType.unexpected.name().toString();
        message = message;
    }
}
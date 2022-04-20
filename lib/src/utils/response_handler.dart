enum ErrorType { success, warning, error, unexpected }

abstract class ResponseHandler {
  late String type;
  late String message;
  StackTrace? stackTrace;
  dynamic data;
}

class Success extends ResponseHandler {
  Success({required String message, dynamic data}) {
    type = ErrorType.success.name.toString();
    message = message;
    this.data = data;
  }
}

class Error extends ResponseHandler {
  Error({required String message, StackTrace? stackTrace}) {
    type = ErrorType.error.name.toString();
    message = message;
    stackTrace = stackTrace;
  }
}

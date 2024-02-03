import basic

while True:
    text = input('basic ->')
    result, error = basic.run(text, '<stdio>')

    if (error is None) and (result is not None):
        print(result)
    elif result is None and error is not None:
        print(error)

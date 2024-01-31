import basic

while True:
    text = input('basic ->')
    result, error = basic.run(text, '<stdio>')

    if error is None:
        print(result)
    else:
        print(error)

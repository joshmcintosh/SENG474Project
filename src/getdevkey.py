secret_file = "secret.json"


def getdevkey():
    with open(secret_file) as fp:
        key = fp.readline()
    return key

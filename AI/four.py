def sort_by_weight(arr, base):
    for i in range(len(arr)):
        arr[i] = [arr[i], calculate_weight(arr[i], base)]

    arr.sort(key=lambda x: (x[1], arr.index(x))

    res = []
    for item in arr:
        res.append(item[0])
    return res


def calculate_weight(num, base):
    weight = 0
    while num > 0:
        weight += (num % base) * (num % base)
        num //= base
    return weight


if __name__ == "__main__":
    c = int(input())
    for _ in range(c):
        n, b = map(int, input().split())
        array = list(map(int, input().split())
        result = sort_by_weight(array, b)
        print(" ".join(map(str, result)))
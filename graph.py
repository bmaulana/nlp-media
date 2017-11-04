import json
import sys
import dateutil.parser


def get_json_array(path):
    arr = []
    with open(path) as f:
        for line in f:
            arr.append(json.loads(line))
    return arr


def main():
    path = sys.argv[1] + '.json'
    articles = get_json_array(path)

    datetimes = []
    for a in articles:
        raw = list(a.values())[0]['datetime']
        datetimes.append(dateutil.parser.parse(raw))

    dates = {}
    for d in datetimes:
        date = d.date()
        if date in dates:
            dates[date] += 1
        else:
            dates[date] = 1

    print(dates)


if __name__ == '__main__':
    main()

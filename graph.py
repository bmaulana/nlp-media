import json
import sys
import dateutil.parser
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

    if len(sys.argv) > 2 and sys.argv[2] == 'day':
        days = {}
        earliest = datetime.date.today()
        for d in datetimes:
            day = d.date()
            if day < earliest:
                earliest = day
            if day in days:
                days[day] += 1
            else:
                days[day] = 1

        while earliest < datetime.date.today():
            if earliest not in days:
                days[earliest] = 0
            earliest += datetime.timedelta(days=1)

        keys = list(days.keys())
        keys.sort()

        print(days)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(keys, [days[x] for x in keys])
        plt.gcf().autofmt_xdate()
        plt.xlabel('Day')

    elif len(sys.argv) > 2 and sys.argv[2] == 'week':
        weeks = {}
        earliest = datetime.date.today()
        for d in datetimes:
            week = d.date() - datetime.timedelta(days=d.date().isocalendar()[2]-1)
            if week < earliest:
                earliest = week
            if week in weeks:
                weeks[week] += 1
            else:
                weeks[week] = 1

        while earliest < datetime.date.today():
            if earliest not in weeks:
                weeks[earliest] = 0
            earliest += datetime.timedelta(days=7)

        keys = list(weeks.keys())
        keys.sort()

        print(weeks)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(0))
        plt.plot(keys, [weeks[x] for x in keys])
        plt.gcf().autofmt_xdate()
        plt.xlabel('Week starting')

    elif len(sys.argv) > 2 and sys.argv[2] == 'month':
        months = {}
        earliest = datetime.date.today()
        for d in datetimes:
            month = d.date().replace(day=1)
            if month < earliest:
                earliest = month
            if month in months:
                months[month] += 1
            else:
                months[month] = 1

        while earliest < datetime.date.today():
            if earliest not in months:
                months[earliest] = 0
            if earliest.month is not 12:
                earliest = datetime.date(earliest.year, earliest.month + 1, 1)
            else:
                earliest = datetime.date(earliest.year + 1, 1, 1)

        keys = list(months.keys())
        keys.sort()

        print(months)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.plot(keys, [months[x] for x in keys])
        plt.gcf().autofmt_xdate()
        plt.xlabel('Month')

    else:
        return  # do not show graph

    plt.ylabel('#articles')
    plt.show()


if __name__ == '__main__':
    main()

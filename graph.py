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
        for d in datetimes:
            day = d.date()
            if day in days:
                days[day] += 1
            else:
                days[day] = 1

        print(days)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(days.keys(), days.values())
        plt.gcf().autofmt_xdate()
        plt.xlabel('Day')

    elif len(sys.argv) > 2 and sys.argv[2] == 'week':
        week_starts = {}
        weeks = {}
        for d in datetimes:
            isocal = d.date().isocalendar()
            week = (isocal[0], isocal[1])
            if week in weeks:
                weeks[week] += 1
            else:
                weeks[week] = 1
                week_starts[week] = d.date() - datetime.timedelta(days=isocal[2]-1)

        # comment out plt.gca()... and plt.gcf()... to show months only
        print([(week_starts[week], weeks[week]) for week in list(weeks.keys())])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(0))
        plt.plot([week_starts[week] for week in list(weeks.keys())], weeks.values())
        plt.gcf().autofmt_xdate()
        plt.xlabel('Week starting')

    elif len(sys.argv) > 2 and sys.argv[2] == 'month':
        months = {}
        for d in datetimes:
            month = d.date().replace(day=1)
            if month in months:
                months[month] += 1
            else:
                months[month] = 1

        print(months)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.plot(months.keys(), months.values())
        plt.gcf().autofmt_xdate()
        plt.xlabel('Month')

    else:
        return  # do not show graph

    plt.ylabel('#articles')
    plt.show()


if __name__ == '__main__':
    main()

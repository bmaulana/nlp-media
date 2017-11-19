import json
import sys
import os


def tokenise(text):
    # TODO clean ASCII characters, tokenise text, (don't remove stop words yet)
    return text


def main():
    if not os.path.exists('./out-filtered/'):
        os.makedirs('./out-filtered/')
    in_path = './out/' + sys.argv[1] + '.json'
    out_path = './out-filtered/' + sys.argv[1] + '.json'
    topic = sys.argv[1].split('-', 1)[1]

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')
    in_lines, out_lines = 0, 0
    for line in f_in:
        in_lines += 1
        text = json.loads(line)
        text = list(text.values())[0]['text']
        text = tokenise(text)

        # TODO do more advanced n-gram filtering
        if topic.lower() in text.lower():
            out_lines += 1
            # TODO tokenise output (use json.dumps(text))
            f_out.write(line)

    print('Input length =', in_lines, 'Output length =', out_lines)
    f_in.close()
    f_out.close()


if __name__ == '__main__':
    main()

import sys
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt


def eval_sentiment(num_samples=5):
    # for each label, print the mean score of sentences I manually class positive, neutral, or negative
    # issue: bias in my classification. Mention in report, further work could involve having multiple reviewers
    labels = ['openai', 'vader', 'xiaohan', 'kcobain', 'stanford', 'textblob', 'textblob_bayes']
    mean_scores = np.array([[0.0, 0.0, 0.0]] * len(labels))
    all_scores = [([], [], []) for i in range(len(labels))]
    answers = []  # to save answers so results are reproducible
    num_sents = 0

    if not os.path.exists('./out-eval/'):
        os.makedirs('./out-eval/')

    # From each file in ./out-sentiment/, show a sample of num_sents sentences to manually classify
    f_out = open('./out-eval/eval.json', 'w', encoding='utf-8')
    for filename in os.listdir('./out-sentiment'):
        f_in = open('./out-sentiment/' + filename, 'r', encoding='utf-8')
        sents = {}
        for line in f_in:
            try:
                raw = json.loads(line)
                for sent, scores in raw['relevant_sentences'].items():
                    if scores['sentiment_score_openai'] != 'ERROR':  # if one score is 'ERROR', all is 'ERROR'
                        sents[sent] = scores
            except (json.JSONDecodeError, TypeError):
                continue
        f_in.close()

        if len(sents) > num_samples:
            sampled = random.Random(0).sample(list(sents.items()), k=num_samples)
        elif len(sents) == 0:  # file read errors
            continue
        else:
            sampled = list(sents.items())

        for sent, scores in sampled:
            res = input(sent + '\n')
            print(scores)
            print()
            if res == '+':
                # positive
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    current_score = max(-1.0, min(1.0, float(scores[full_label])))
                    mean_scores[i, 0] += float(scores[full_label])
                    all_scores[i][0].append(current_score)
                answers.append((sent, '+'))
            elif res == '-':
                # negative
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    current_score = max(-1.0, min(1.0, float(scores[full_label])))
                    mean_scores[i, 2] += float(scores[full_label])
                    all_scores[i][2].append(current_score)
                answers.append((sent, '-'))
            elif res == 'o':
                # out of topic
                continue
            else:
                # neutral
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    current_score = max(-1.0, min(1.0, float(scores[full_label])))
                    mean_scores[i, 1] += float(scores[full_label])
                    all_scores[i][1].append(current_score)
                answers.append((sent, 'n'))
            num_sents += 1

            to_write = {'sentence': sent, 'label': answers[-1][1]}
            for i in range(len(labels)):
                full_label = 'sentiment_score_' + labels[i]
                to_write[full_label] = scores[full_label]
            f_out.write(json.dumps(to_write))
            f_out.write('\n')
    f_out.close()
    json_to_list('./out-eval/eval.json')

    # Print mean scores for each classification, for each sentiment scorer
    mean_scores /= num_sents
    for i in range(len(labels)):
        print(labels[i], '\tPositive:', mean_scores[i][0], '\tNeutral:', mean_scores[i][1],
              '\tNegative:', mean_scores[i][2])

    # output sentences and answers for reproducibility
    f_out = open('./out-eval/eval.csv', 'w', encoding='utf-8')
    for answer in answers:
        f_out.write(str(answer[1]) + ';' + str(answer[0]) + '\n')
    f_out.close()

    # Plot histogram for each classification, for each sentiment scorer
    fig, axs = plt.subplots(2, (len(labels) + 1) // 2, figsize=(5 * ((len(labels) + 1) // 2), 10), tight_layout=True)
    for i in range(len(labels)):
        ax = axs[i % 2, i // 2]
        ax.hist(all_scores[i][0], bins=np.arange(-1.0, 1.2, 0.1), alpha=0.5, label='pos')
        ax.hist(all_scores[i][2], bins=np.arange(-1.0, 1.2, 0.1), alpha=0.5, label='neg')
        ax.hist(all_scores[i][1], bins=np.arange(-1.0, 1.2, 0.1), alpha=0.3, label='ntr')
        ax.set_title(labels[i])
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('No. of articles')
        ax.legend()
    plt.savefig('./out-eval/eval.png')
    plt.close()


# Converts a JSON-object-in-each-line file to a JSON array file
def json_to_list(fname):
    f_in = open(fname, 'r', encoding='utf-8')
    json_list = []
    for line in f_in:
        json_list.append(json.loads(line))
    f_in.close()
    f_out = open(fname, 'w', encoding='utf-8')
    f_out.write(json.dumps(json_list))
    f_out.close()
    pass


# function that reads a labelled JSON file with +/-/n and scores for each scorer
def eval_sentiment_read(fname):
    f_in = open(fname, 'r', encoding='utf-8')
    data = json.load(f_in)
    f_in.close()

    # for each label, print the mean score of sentences I manually class positive, neutral, or negative
    # issue: bias in my classification. Mention in report, further work could involve having multiple reviewers
    labels = ['vader', 'xiaohan', 'kcobain', 'openai', 'stanford', 'textblob', 'textblob_bayes']
    all_scores = [([], [], []) for i in range(len(labels))]  # (positive, neutral, negative)
    answers = []  # to save answers so results are reproducible

    if not os.path.exists('./out-eval/'):
        os.makedirs('./out-eval/')

    for sent in data:
        res = sent['label']
        if res == '+':
            # positive
            for i in range(len(labels)):
                full_label = 'sentiment_score_' + labels[i]
                current_score = max(-1.0, min(1.0, float(sent[full_label])))
                all_scores[i][0].append(current_score)
            answers.append((sent['sentence'], 'positive'))
        elif res == '-':
            # negative
            for i in range(len(labels)):
                full_label = 'sentiment_score_' + labels[i]
                current_score = max(-1.0, min(1.0, float(sent[full_label])))
                all_scores[i][2].append(current_score)
            answers.append((sent['sentence'], 'negative'))
        elif res == 'n':
            # neutral
            for i in range(len(labels)):
                full_label = 'sentiment_score_' + labels[i]
                current_score = max(-1.0, min(1.0, float(sent[full_label])))
                all_scores[i][1].append(current_score)
            answers.append((sent['sentence'], 'neutral'))
        elif res == 'o':
            # out of topic
            answers.append((sent['sentence'], 'irrelevant'))

    # output sentences and answers in txt file with no scores, to peer-review my labels
    f_out = open('./out-eval/eval.txt', 'w', encoding='utf-8')
    for answer in answers:
        f_out.write(str(answer[1]) + '; ' + str(answer[0]) + '\n')

    f_out.write('\n')

    for i in range(len(labels)):
        pos, neu, neg = np.array(all_scores[i][0]), np.array(all_scores[i][1]), np.array(all_scores[i][2])
        f_out.write(labels[i] + ':\n')
        f_out.write('\tMean Positive: ' + str(np.average(pos)) + '\n')
        f_out.write('\tMean Neutral: ' + str(np.average(neu)) + '\n')
        f_out.write('\tMean Negative: ' + str(np.average(neg)) + '\n')
        f_out.write('\tTrue Positive: ' + str(pos[pos > 0.0].shape[0]) + '\n')
        f_out.write('\tFalse Positive: ' + str(neg[neg > 0.0].shape[0]) + '\n')
        f_out.write('\tTrue Negative: ' + str(neg[neg <= 0.0].shape[0]) + '\n')
        f_out.write('\tFalse Negative: ' + str(pos[pos <= 0.0].shape[0]) + '\n')
        f_out.write('\tAccuracy: ' + str((pos[pos > 0.0].shape[0] + neg[neg <= 0.0].shape[0]) /
                                         (pos.shape[0] + neg.shape[0])) + '\n')
    f_out.close()

    # Plot histogram for each classification, for each sentiment scorer
    fig, axs = plt.subplots(2, (len(labels) + 1) // 2, figsize=(5 * ((len(labels) + 1) // 2), 10), tight_layout=True)
    for i in range(len(labels)):
        ax = axs[i % 2, i // 2]
        ax.hist(all_scores[i][0], bins=np.arange(-1.0, 1.2, 0.1), alpha=0.4, label='pos')
        ax.hist(all_scores[i][2], bins=np.arange(-1.0, 1.2, 0.1), alpha=0.4, label='neg')
        ax.hist(all_scores[i][1], bins=np.arange(-1.0, 1.2, 0.1), alpha=0.2, label='ntr')
        ax.set_title(labels[i])
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('No. of articles')
        ax.legend()
    plt.savefig('./out-eval/eval.png')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'tolist':
            json_to_list('./out-eval/eval.json')
        elif sys.argv[1] == 'read':
            eval_sentiment_read('./out-eval/eval.json')
        else:
            eval_sentiment(int(sys.argv[1]))
    else:
        eval_sentiment()

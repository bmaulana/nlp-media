import sys
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt


def eval_sentiment(num_samples=5):
    # for each label, print the mean score of sentences I manually class positive, neutral, or negative
    # issue: bias in my classification. Mention in report, further work could involve having multiple reviewers
    labels = ['vader', 'xiaohan', 'kcobain', 'openai', 'stanford', 'textblob', 'textblob_bayes']
    mean_scores = np.array([[0.0, 0.0, 0.0]] * len(labels))
    all_scores = [([], [], []) for i in range(len(labels))]
    answers = []  # to save answers so results are reproducible
    out_file = './out-sentiment/eval.csv'
    num_sents = 0

    # From each file in ./out-sentiment/, show a sample of num_sents sentences to manually classify
    for filename in os.listdir('./out-sentiment'):
        f_in = open('./out-sentiment/' + filename, 'r')
        sents = {}
        for line in f_in:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sent, scores in raw['relevant_sentences'].items():
                if scores['sentiment_score_openai'] != 'ERROR':  # if one score is 'ERROR', all is 'ERROR'
                    sents[sent] = scores
        f_in.close()

        if len(sents) > num_samples:
            sampled = random.Random(0).sample(list(sents.items()), k=num_samples)
        elif len(sents) == 0:  # decode error
            continue
        else:
            sampled = list(sents.items())

        for sent, scores in sampled:
            res = input(sent + '\n')
            print(scores)
            print()
            if res == '+':
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    mean_scores[i, 0] += float(scores[full_label])
                    all_scores[i][0].append(float(scores[full_label]))
                answers.append((sent, '+'))
            elif res == '-':
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    mean_scores[i, 2] += float(scores[full_label])
                    all_scores[i][2].append(float(scores[full_label]))
                answers.append((sent, '-'))
            else:
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    mean_scores[i, 1] += float(scores[full_label])
                    all_scores[i][1].append(float(scores[full_label]))
                answers.append((sent, 'n'))
            num_sents += 1

    # Print mean scores for each classification, for each sentiment scorer
    mean_scores /= num_sents
    for i in range(len(labels)):
        print(labels[i], '\tPositive:', mean_scores[i][0], '\tNeutral:', mean_scores[i][1],
              '\tNegative:', mean_scores[i][2])

    # output sentences and answers for reproducibility
    f_out = open(out_file, 'w')
    for answer in answers:
        f_out.write(str(answer[1]) + ',' + str(answer[0]) + '\n')
    f_out.close()

    # Plot histogram for each classification, for each sentiment scorer
    fig, axs = plt.subplots(2, (len(labels) + 1) // 2, figsize=(5 * ((len(labels) + 1) // 2), 10), tight_layout=True)
    for i in range(len(labels)):
        ax = axs[i % 2, i // 2]
        ax.hist(all_scores[i][0], bins=np.arange(-1.0, 1.1, 0.1), alpha=0.5, label='pos')
        ax.hist(all_scores[i][2], bins=np.arange(-1.0, 1.1, 0.1), alpha=0.5, label='neg')
        ax.hist(all_scores[i][1], bins=np.arange(-1.0, 1.1, 0.1), alpha=0.3, label='ntr')
        ax.set_title(labels[i])
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('No. of articles')
        ax.legend()
    plt.savefig('./out-plot/sentiment_eval.png')
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        eval_sentiment(int(sys.argv[1]))
    eval_sentiment()

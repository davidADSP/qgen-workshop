import matplotlib.pyplot as plt
import numpy as np

from qgen.embedding import look_up_token, look_up_word, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN, END_WORD

def show_losses(training_loss_history, test_loss_history):
        plt.plot(training_loss_history)
        plt.plot(test_loss_history)
        plt.show()


def show_test(batch, answer_batch, idx, answer_model, decoder_initial_state_model, question_model):
        print('\n TEXT')
        snippet_length = min(len(batch['document_text'][idx]), 1500)
        print(batch['document_text'][idx][:snippet_length])

        # print('\n LAST WORDS')
        # print(batch['document_words'][idx][-5:])

        # print('\n QUESTION TEXT')
        # print(batch['question_text'][idx])

        # print('\n ANSWER TEXT')
        # print(batch['answer_text'][idx])

        # print('\n ANSWER INDICES')
        # ai = batch['answer_indices'][idx]
        # print(ai)

        # print('\n ANSWER TOKENS')
        # print(batch['document_tokens'][idx][ai])


        print('\n PREDICTED ANSWER TEXT')
        print(answer_batch['answer_text'][idx])

        print('\n PREDICTED ANSWER INDICES')
        print(answer_batch['answer_indices'][idx])

        if len(answer_batch['answer_indices'][idx]) > 0:


                next_decoder_init_state = decoder_initial_state_model.predict([answer_batch['document_tokens'][[idx]], answer_batch['answer_masks'][[idx]]])

                word_tokens = [START_TOKEN]
                questions = [look_up_token(START_TOKEN)]

                ended = False

                while not ended:

                        word_preds, next_decoder_init_state = question_model.predict([word_tokens, next_decoder_init_state])

                        next_decoder_init_state = np.squeeze(next_decoder_init_state, axis = 1)
                        word_tokens = np.argmax(word_preds, 2)[0]

                        questions.append(look_up_token(word_tokens[0]))

                        if word_tokens[0] == END_TOKEN:
                                ended = True

                questions = ' '.join(questions)

                print('\n MODEL QUESTION TEXT')
                print(questions)




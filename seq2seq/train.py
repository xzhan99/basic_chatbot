from seq2seq.build_model import train_seq2seq_model, build_seq2seq_model, save_seq2seq_model
from seq2seq.preprocessing import build_dataset

if __name__ == '__main__':
    dataset = build_dataset()  # the dict will be used in section 3 for training
    print(dataset['comic']['training_data'][:10])

    # testing: Generate a batch data
    input_batch, output_batch, target_batch = make_batch(dataset['professional'], sg_model)
    print(input_batch[0].shape)
    print(output_batch[0].shape)
    print(target_batch[0].shape)

    # testing
    enc_input, dec_input, targets, model, cost, optimizer = build_seq2seq_model(
        len(dataset['professional']['answer_dict']))
    sess = train_seq2seq_model(enc_input, dec_input, targets, cost, optimizer, input_batch, output_batch, target_batch)

    # save session
    save_seq2seq_model(sess, '../model/testing_model.ckpt')

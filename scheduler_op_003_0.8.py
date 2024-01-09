import os

if __name__ == '__main__':

    os.system('python main_proposed_model_op.py --dataset=FD003 --label_size=0.8 --save_place=result_003_label0.8.csv')
    os.system('python main_compare_bi_lstm_attention_op.py --dataset=FD003 --label_size=0.8 --save_place=result_003_label0.8.csv')
    os.system('python main_compare_double_attention.py --dataset=FD003 --label_size=0.8 --save_place=result_003_label0.8.csv')
    os.system('python main_compare_gru_op.py --dataset=FD003 --label_size=0.8 --save_place=result_003_label0.8.csv')
    os.system('python main_compare_lstm_op.py --dataset=FD003 --label_size=0.8 --save_place=result_003_label0.8.csv')
    os.system('python main_compare_resnet_op.py --dataset=FD003 --label_size=0.8 --save_place=result_003_label0.8.csv')
    os.system('python main_compare_vision_transformer_op.py --dataset=FD003 --label_size=0.8 --save_place=result_003_label0.8.csv')
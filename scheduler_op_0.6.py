import os

if __name__ == '__main__':

    os.system(
        'python main_proposed_model_op.py --dataset=FD001 --label_size=0.6 --save_place=result_label0.6.csv')
    os.system('python main_compare_bi_lstm_attention_op.py --dataset=FD001 --label_size=0.6 --save_place=result_label0.6.csv')
    os.system('python main_compare_double_attention_op.py --dataset=FD001 --label_size=0.6 --save_place=result_label0.6.csv')
    os.system(
        'python main_compare_gru_op.py --dataset=FD001 --label_size=0.6 --save_place=result_label0.6.csv')
    os.system(
        'python main_compare_lstm_op.py --dataset=FD001 --label_size=0.6 --save_place=result_label0.6.csv')
    os.system(
        'python main_compare_resnet_op.py --dataset=FD001 --label_size=0.6 --save_place=result_label0.6.csv')
    os.system('python main_compare_vision_transformer_op.py --dataset=FD001 --label_size=0.6 --save_place=result_label0.6.csv')

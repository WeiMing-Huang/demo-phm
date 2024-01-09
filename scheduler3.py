import os

if __name__ == '__main__':

    os.system(
        'python main_proposed_model.py --dataset=FD003 --save_place=result3.csv')
    os.system(
        'python main_compare_bi_lstm_attention.py --dataset=FD003 --save_place=result3.csv')
    os.system(
        'python main_compare_bi_lstm.py --dataset=FD003 --save_place=result3.csv')
    os.system(
        'python main_compare_double_attention.py --dataset=FD003 --save_place=result3.csv')
    os.system('python main_compare_gru.py --dataset=FD003 --save_place=result3.csv')
    os.system(
        'python main_compare_lstm_transformer.py --dataset=FD003 --save_place=result3.csv')
    os.system('python main_compare_lstm.py --dataset=FD003 --save_place=result3.csv')
    os.system(
        'python main_compare_resnet.py --dataset=FD003 --save_place=result3.csv')
    os.system(
        'python main_compare_timeseries_transformer.py --dataset=FD003 --save_place=result3.csv')
    os.system(
        'python main_compare_vision_transformer.py --dataset=FD003 --save_place=result3.csv')

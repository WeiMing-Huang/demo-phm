import os

if __name__ == '__main__':

    os.system('python main_proposed_model_op.py --dataset=FD001 --label_size=0.01 --save_place=result_label.csv')
    os.system('python main_proposed_model_op.py --dataset=FD001 --label_size=0.05 --save_place=result_label.csv')
    os.system('python main_proposed_model_op.py --dataset=FD001 --label_size=0.1 --save_place=result_label.csv')



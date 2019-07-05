source ~/.bashrc

# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng' 'train'
# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc' 'train'
# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc' 'train'
# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng' 'train'


# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng' 'test_seq'
# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc' 'test_seq'
# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc' 'test_seq'
# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng' 'test_seq'


#python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc' 'test_seq_other' '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng'
#rm -rf /esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc/MPI_CG_test_seq_other_data_marc2weipeng
#mv /esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc/MPI_CG_test_seq_other_data /esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc/MPI_CG_test_seq_other_data_marc2weipeng
python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc' 'test_seq_other' '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng'
rm -rf /esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc/MPI_CG_test_seq_other_data_marc2weipeng
mv /esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc/MPI_CG_test_seq_other_data /esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc/MPI_CG_test_seq_other_data_marc2weipeng
# python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng' 'test_seq_other' '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc'
# rm -rf /esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng/MPI_CG_test_seq_other_data_weipeng2marc
# mv /esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng/MPI_CG_test_seq_other_data /esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng/MPI_CG_test_seq_other_data_weipeng2marc
python convert_RCV.py '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng' 'test_seq_other' '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc'
rm -rf /esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng/MPI_CG_test_seq_other_data_weipeng2marc
mv /esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng/MPI_CG_test_seq_other_data /esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng/MPI_CG_test_seq_other_data_weipeng2marc

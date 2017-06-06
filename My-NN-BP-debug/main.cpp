#include "NNet.h"
#include <fstream>
#include <sstream>

void ReadSample(sample * sampleInOut);
int ReverseInt (int i);

int main()
{
    NNet testNet;
    sample sampleInOut[100];

    ReadSample(sampleInOut);

    vector<sample> sampleGroup(sampleInOut, sampleInOut+100);
    
    testNet.training(sampleGroup, 0.05);

    testNet.predict(sampleGroup);
    
    return 0;
}

void ReadSample(sample * sampleInOut)
{
    ifstream s_file ("samples.idx3-ubyte", ios::binary);
    ifstream l_file ("labels.idx1-ubyte", ios::binary);

    if (s_file.is_open() && l_file.is_open())
    {
        int magic_number=0;
        int l_magic_number=0;
        int number_of_images=0;
        int l_number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        s_file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        cout << "magic_number: " << magic_number << endl;

        l_file.read((char*)&l_magic_number,sizeof(l_magic_number));
        l_magic_number= ReverseInt(l_magic_number);
        cout << "magic_number: " << l_magic_number << endl;

        s_file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        cout << "number_of_images: " << number_of_images << endl;
        
        l_file.read((char*)&l_number_of_images,sizeof(l_number_of_images));
        l_number_of_images= ReverseInt(l_number_of_images);
        cout << "number_of_images: " << l_number_of_images << endl;

        s_file.read((char*)&n_rows,sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        cout << "n_rows: " << n_rows << endl;
        
        s_file.read((char*)&n_cols,sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        cout << "n_cols: " << n_cols << endl;

        for (int index = 0; index < number_of_images; index++)
        {
            uint8_t temp_label;

            l_file.read((char*)&temp_label, sizeof(temp_label));
            cout << (int)temp_label << endl;

            for (int i = 0; i < n_rows; i++)
            {
                for (int j = 0; j < n_cols; j++)
                {
                    uint8_t temp;
                    s_file.read((char*)&temp,sizeof(temp));
                    cout << (int)temp;
                    sampleInOut[index].in.push_back((int)temp);
                }
                cout << endl;
            }

            for (int i = 0; i < 10; i++)
            {
                if ((int)temp_label == i)
                    sampleInOut[index].out.push_back(1);
                else
                    sampleInOut[index].out.push_back(0);
            }
        }
    }
    s_file.close();
    l_file.close();
}

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
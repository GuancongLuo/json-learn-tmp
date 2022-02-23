#include "json.hpp"

using json = nlohmann::json;

#include <iostream>
#include <fstream>
#include <string>

int main()
{
    json j; 

    std::string filename = "../src/config.json";

    std::fstream s(filename);
    if (!s.is_open())
    {
        std::cout << "failed to open " << filename << '\n';
    }
    else
    {
        s >> j;
    }

    std::cout << j << std::endl;
    std::cout << j["is_only_get_point_cloud"] << std::endl;
    if (j["is_only_get_point_cloud"])
    {
        return 0;
    }
    
    std::cout << j["point_cloud"] << std::endl;

    std::cout << (j["pointcloud2"]["point_cloud_name"]) << std::endl;
    
    // char file_name1[1024];
    // char fullpath1[1024];
    // char file_name2[1024];
    // char fullpath2[1024];


    std::string data_dir = j["Dataset_dir_abs_path"];
    std::string file_name1 = j["pointcloud2"]["file_name"];
    std::string dir = data_dir + file_name1;

    std::cout << data_dir << std::endl;
    std::cout << file_name1 << std::endl;
    std::cout << dir << std::endl;

    std::vector<float> camera(4);
    float scale = 1;
    camera[0] = j["camera"]["fx"].get<float>(); //fx
    camera[1] = 532.740352; //fy
    camera[2] = 640; //cx
    camera[3] = 380;//cy

    for(float f : camera){
        std::cout <<"camera params: "<< f << std::endl;
    }


    return 0;
}
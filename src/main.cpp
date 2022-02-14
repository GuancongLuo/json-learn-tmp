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
    std::cout << j["point_cloud"] << std::endl;

    for(json point_cloud : j["point_cloud"]){
        std::cout << point_cloud["point_cloud_name"] << std::endl;
        std::cout << point_cloud["star_address"] << std::endl;
        std::cout << point_cloud["point_cloud_name"].is_number() << std::endl;
    }
    
    return 0;
}
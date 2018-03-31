#ifndef DATASET_H
#define DATASET_H
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

namespace ml {
    using Index = Eigen::Index;
    using Eigen::Dynamic;
    using Eigen::RowMajor;

    template <typename FloatType>
    using DataSet = Eigen::Array<FloatType, Dynamic, Dynamic, RowMajor>;

    template <typename LabelType>
    using Labels = Eigen::Array<LabelType, Dynamic, 1>;

    template <typename FeatureType>
    void Normalize(std::shared_ptr<DataSet<FeatureType>> Xptr)
    {
        auto& X = *Xptr;
        auto m = X.rows();
        using namespace std;


        cout << m << ' ' << X.cols() << endl;
        Eigen::Array<FeatureType, 1, Eigen::Dynamic> mins = X.matrix().colwise().minCoeff();
        Eigen::Array<FeatureType, 1, Eigen::Dynamic> maxs = X.matrix().colwise().maxCoeff();
        cout << "fine" << endl;
        auto ranges = maxs - mins;
        X -= mins.colwise().replicate(m);
        DataSet<FeatureType> repRanges = ranges.colwise().replicate(m);
        X /= repRanges;

    }

    template <typename FeatureType=float>
    std::pair<std::shared_ptr<DataSet<FeatureType>>, 
              std::shared_ptr<Labels<FeatureType>>> LoadIris()
    {
        using Vector4 = Eigen::Matrix<FeatureType, 1, 4>;
        std::vector<Vector4, Eigen::aligned_allocator<Vector4>> rows;
        std::vector<FeatureType> temp_labels;
        std::ifstream iris_file("../datasets/iris.csv");
        std::string buffer;
        std::unordered_map<std::string, FeatureType> label_map;
        int label_index = 0;

        while (std::getline(iris_file, buffer)) {
            std::stringstream line_stream(buffer);
            std::string label;
            Vector4 row;
            FeatureType a, b , c, d;
            char comma;

            line_stream >> a >> comma
                        >> b >> comma
                        >> c >> comma
                        >> d >> comma
                        >> label;
            row << a, b, c, d;

            if (label_map.find(label) == label_map.end())
                label_map[label] = ++label_index;

            rows.push_back(row);
            temp_labels.push_back(label_map[label]);
        }

        Index size = rows.size();
        auto X = std::make_shared<DataSet<FeatureType>>(size, 4);
        auto Y = std::make_shared<Labels<FeatureType>>(size);

        for (Index i = 0; i != size; ++i) {
            X->row(i) = rows[i];
            (*Y)(i) = temp_labels[i];
        }
        
        return std::make_pair(X, Y);
    }
}

#endif // DATASET_H

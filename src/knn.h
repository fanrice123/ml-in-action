#ifndef KNN_H
#define KNN_H
#include <type_traits>
#include <memory>
#include <vector>
#include "dataset.h"

namespace ml {

    template <typename FeatureType>
    class KNNClassifier {
    public:
        KNNClassifier(std::shared_ptr<DataSet<FeatureType>> features, 
                      std::shared_ptr<Labels<FeatureType>> labels, 
                      unsigned k_val);

        Labels<FeatureType> Classify(const DataSet<FeatureType>& feature);

    private:
        std::shared_ptr<DataSet<FeatureType>> X;
        std::shared_ptr<Labels<FeatureType>> Y;
        unsigned k;

        template <typename BlockType>
        FeatureType find_top_vote_from_k(BlockType&& dists) const;
    };

    template <typename FeatureType, 
              typename... Args,
              typename=std::enable_if_t<std::is_floating_point_v<FeatureType>>>
    KNNClassifier<FeatureType> CreateKNN(std::shared_ptr<DataSet<FeatureType>> features,
                                         std::shared_ptr<Labels<FeatureType>> labels,
                                         Args&&... args)
    {
        return KNNClassifier<FeatureType>(features, 
                                                     labels, 
                                                     std::forward<Args>(args)...);
    }

}

#include "include/knn_impl.h"

#endif // KNN_H

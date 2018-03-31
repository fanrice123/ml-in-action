#include <utility>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <unordered_map>
#include <vector>
#include <queue>

namespace {
    template <typename FeatureType>
    struct pair_hash {
        std::size_t operator()(const std::pair<FeatureType, FeatureType>& pair) const noexcept
        {
            return hasher(pair.first);
        }
        std::hash<FeatureType> hasher;
    };

}

namespace ml {
    template <typename FeatureType>
    KNNClassifier<FeatureType>::
    KNNClassifier(std::shared_ptr<DataSet<FeatureType>> features, 
                  std::shared_ptr<Labels<FeatureType>> labels, 
                  unsigned k_val)
        : X(features), Y(labels), k(k_val) {}

    template <typename FeatureType>
    Labels<FeatureType>
    KNNClassifier<FeatureType>::Classify(const DataSet<FeatureType>& features)
    {
        const Index X_m = X->rows(), features_m = features.rows();
        Labels<FeatureType> predictions(features_m);

        auto features_m_ones = Eigen::Matrix<FeatureType, 
                                                           Eigen::Dynamic, 
                                                           1>::Ones(X_m);
        Eigen::Matrix<FeatureType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp_features = features.matrix();
        DataSet<FeatureType> repFeatures = Eigen::kroneckerProduct(temp_features, 
                                                                 features_m_ones).eval().array();

        DataSet<FeatureType> repX = X->replicate(features_m, 1);
        Labels<FeatureType> dists = (repFeatures - repX).square().rowwise().sum().sqrt();

        for (Index i = 0; i != features_m; ++i) {
            predictions[i] = find_top_vote_from_k(dists.block(i * X_m, 0, X_m, 1));
        }

        return predictions;
    }

    template <typename FeatureType>
    template <typename BlockType>
    FeatureType KNNClassifier<FeatureType>::find_top_vote_from_k(BlockType&& dists) const
    {
        typedef std::pair<FeatureType, FeatureType> DistLabelPair;
        auto dist_comp = [] (const DistLabelPair& lhs, const DistLabelPair& rhs) {
            return lhs.first > rhs.first;
        };

        Index X_m = X->rows();
        std::vector<DistLabelPair> m_vec;
        m_vec.reserve(X_m);
        // top_k
        std::priority_queue<DistLabelPair,
                            std::vector<DistLabelPair>,
                            decltype(dist_comp)> pq{dist_comp, std::move(m_vec)};

        for (Index i = 0; i != X_m; ++i)
            pq.push({dists(i, 0), (*Y)(i)});

        std::unordered_map<DistLabelPair, unsigned, pair_hash<FeatureType>> vote_count;

        for (unsigned i = 0; i != k; ++i) {
            ++vote_count[pq.top()];
            pq.pop();
        }

        FeatureType top_feature;
        unsigned top_vote = 0;
        for (auto& [feature, vote] : vote_count) {
            if (vote > top_vote) {
                top_vote = vote;
                top_feature = feature.second;
            }
        }
        
        return top_feature;
    }
}

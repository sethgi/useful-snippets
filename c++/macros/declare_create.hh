#define DECLARE_CREATE(ClassName, ...)                                   \
public:                                                                  \
    template <typename... Args>                                          \
    static std::shared_ptr<ClassName> Create(Args&&... args) {           \
        return std::make_shared<ClassName>(std::forward<Args>(args)...); \
    }
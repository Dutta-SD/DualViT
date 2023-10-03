def SplitViTHierarchicalTPVitHalfPretrained(SplitHierarchicalTPViT):

    def random_init(self, m):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

    def init_broad(self):
        broad_modules = self.encoder.layer[: self.n_broad]

        for layer in broad_modules:
            self.random_init(layer)

    def forward(self, pixel_values: torch.Tensor):
        super(self).forward()
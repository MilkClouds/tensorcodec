fn main() {
    let mut build = cc::Build::new();
    for library in ["libjpeg", "libwebp", "libwebpdemux", "libavif"] {
        let lib = pkg_config::Config::new()
            .probe(library)
            .unwrap_or_else(|e| {
                panic!("{library}: {e}; run scripts/build_image_deps.sh and set PKG_CONFIG_PATH")
            });
        for path in lib.include_paths {
            build.include(path);
        }
    }
    build.include("vendor/giflib").file("src/image_codecs.c");
    for source in [
        "dgif_lib.c",
        "gifalloc.c",
        "gif_hash.c",
        "openbsd-reallocarray.c",
    ] {
        build.file(format!("vendor/giflib/{source}"));
    }
    build
        .flag_if_supported("-std=gnu11")
        .compile("image_codecs");
    println!("cargo:rerun-if-changed=src/image_codecs.c");
    println!("cargo:rerun-if-changed=vendor/giflib");
}

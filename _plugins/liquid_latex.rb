require "digest"
require "fileutils"

module Jekyll
  module Tags
    class LatexBlock < Liquid::Block
      include Liquid::StandardFilters

      @@globals = {
        "debug" => true,
        "density" => "300",
        "usepackages" => "",
        "latex_cmd" => "pdflatex -interaction=nonstopmode -shell-escape $texfile > /dev/null 2>&1",
        "convert_cmd" => "convert -density $density $pdffile $pngfile > /dev/null 2>&1",
        "optimize_cmd" => "optipng -o7 $pngfile > /dev/null 2>&1",
        "temp_filename" => "tmp_latex_render",
        "output_directory" => "/latex",
        "src_dir" => "",
        "dst_dir" => "",
        "class" => "inline",
      }

      @@generated_files = [ ]
      def self.generated_files
        @@generated_files
      end

      def self.latex_output_directory
        @@globals["output_directory"]
      end

      def initialize(tag_name, text, tokens)
        super
        # We now can adquire the options for this liquid tag
        @p = { "usepackages" => "" }
        text.gsub("  ", " ").split(" ").each do |part|
          if part.split("=").count != 2
            raise SyntaxError.new("Syntax Error in tag 'latex'")
          end
          var,val = part.split("=")
          @p[var] = val
        end
      end

      def self.read_config(name, site)
        cfg = site.config["liquid_latex"]
        return if cfg.nil?
        value = cfg[name]
        @@globals[name] = value if !value.nil?
      end

      def self.init_globals(site)
        # Get all the variables from the config and remember them for future use.
        if !defined?(@@first_time)
          @@first_time = true
          @@globals.keys.each do |k|
            read_config(k, site)
          end
          @@globals["src_dir"] = File.join(site.config["source"], @@globals["output_directory"])
          @@globals["dst_dir"] = File.join(site.config["destination"], @@globals["output_directory"])
          # Verify and prepare the output folder if it doesn't exist
          FileUtils.mkdir_p(@@globals["src_dir"]) unless File.exists?(@@globals["src_dir"])
        end
      end

      def execute_cmd(cmd)
        cmd = cmd.gsub("\$density", @p["density"].to_s)
        cmd = cmd.gsub("\$texfile", @p["tex_fn"])
        cmd = cmd.gsub("\$pdffile", @p["pdf_fn"])
        cmd = cmd.gsub("\$pngfile", @p["png_fn"])
        puts cmd if @@globals["debug"]
        system(cmd)
        return ($?.exitstatus == 0)
      end

      def render(context)
        latex_source = super
        # fix initial configurations
        site = context.registers[:site]
        Tags::LatexBlock::init_globals(site)
        # prepare density and usepackages
        @p["class"] = @@globals["class"] unless @p.key?("class")
        @p["density"] = @@globals["density"] unless @p.key?("density")
        @p["usepackages"] = (@@globals["usepackages"].split(",") + @p["usepackages"].split(",")).join(",")
        # if this LaTeX code is already compiled, skip its compilation
        latex_tex = "\\documentclass[preview]{standalone}\n"
        @p["usepackages"].gsub(" ","").split(",").each do |packagename|
          latex_tex << "\\usepackage\{#{packagename}\}\n"
        end
        latex_tex << "\\usetikzlibrary{shapes,arrows}\n"
        latex_tex << "\\begin{document}\n\\pagestyle{empty}\n"
        latex_tex << latex_source
        latex_tex << "\\end{document}"

        hash_txt = @p["density"].to_s + @p["usepackages"].to_s + latex_tex
        filename = "latex-" + Digest::MD5.hexdigest(hash_txt) + ".png"
        @p["png_fn"] = File.join(@@globals["src_dir"], filename)
        ok = true
        if !File.exists?(@p["png_fn"])
          puts "Compiling with LaTeX..." if @@globals["debug"]
          @p["tex_fn"] = @@globals["temp_filename"] + ".tex"
          @p["pdf_fn"] = @@globals["temp_filename"] + ".pdf"

          # Put the LaTeX source code to file
          tex_file = File.new(@p["tex_fn"], "w")
          tex_file.puts(latex_tex)
          tex_file.close
          # Compile the document to PNG
          ok = execute_cmd(@@globals["latex_cmd"])
          ok = execute_cmd(@@globals["convert_cmd"]) if ok
          ok = execute_cmd(@@globals["optimize_cmd"]) if ok
          # Delete temporary files
          Dir.glob(@@globals["temp_filename"] + "*").each do |f|
            File.delete(f)
          end
        end

        if ok
          # Add the file to the list of static files for the final copy once generated
          st_file = Jekyll::StaticFile.new(site, site.source, @@globals["output_directory"], filename)
          @@generated_files << st_file
          site.static_files << st_file
          # Build the <img> tag to be returned to the renderer
          png_path = File.join(@@globals["output_directory"], filename)
          return "<img class=\"" + @p["class"] + "\" src=\"" + png_path + "\" />"
        else
          # Generate a block of text in the post with the original source
          resp = "<strong style=\"color: red\">Failed to render the following block of LaTeX:</strong><br/>\n"
          resp << "<pre><code>" + latex_tex + "</code></pre>"
          return resp
        end
      end
    end
  end

  class Site
    # Alias for the parent Site::write method (ingenious static override)
    alias :super_latex_write :write

    def write
      super_latex_write   # call the super method
      Tags::LatexBlock::init_globals(self)
      dest_folder = File.join(dest, Tags::LatexBlock::latex_output_directory)
      FileUtils.mkdir_p(dest_folder) unless File.exists?(dest_folder)

      # clean all previously rendered files not rendered in the actual build
      src_files = []
      Tags::LatexBlock::generated_files.each do |f|
        src_files << f.path
      end
      pre_files = Dir.glob(File.join(source, Tags::LatexBlock::latex_output_directory, "latex-*.png"))
      to_remove = pre_files - src_files
      to_remove.each do |f|
        File.unlink f if File.exists?(f)
        d, fn = File.split(f)
        df = File.join(dest, Tags::LatexBlock::latex_output_directory, fn)
        File.unlink df if File.exists?(df)
      end
    end
  end
end

Liquid::Template.register_tag('latex', Jekyll::Tags::LatexBlock)
